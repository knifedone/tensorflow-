在graph中，图的重写的创建是由RewriteGraphForExecution完成的。该函数的实现在tensorflow/core/graph/subgraph.cc中。
首先根据fed_outputs和fetch_outputs创建feed_rewrites和fetch_outputs。然后再调用重载的一个函数RewriteGraphForExecution。
```C++
Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
    const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata) {
  std::vector<std::unique_ptr<PruneRewrite>> feed_rewrites;
  feed_rewrites.reserve(fed_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fed_outputs.size(); ++i) {
      feed_rewrites.emplace_back(new ArgFeedRewrite(
          &fed_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fed_output : fed_outputs) {
      feed_rewrites.emplace_back(
          new RecvFeedRewrite(&fed_output, &device_info));
    }
  }

  std::vector<std::unique_ptr<PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(fetch_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fetch_outputs.size(); ++i) {
      fetch_rewrites.emplace_back(new RetvalFetchRewrite(
          &fetch_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fetch_output : fetch_outputs) {
      fetch_rewrites.emplace_back(
          new SendFetchRewrite(&fetch_output, &device_info));
    }
  }

  return RewriteGraphForExecution(g, feed_rewrites, fetch_rewrites,
                                  target_node_names, out_metadata);
}
```
重载的RewriteGraphForExecution函数的实现为
```C++
Status RewriteGraphForExecution(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    const gtl::ArraySlice<string>& target_node_names,
    RewriteGraphMetadata* out_metadata) {
  if (fetch_rewrites.empty() && target_node_names.empty()) {
    return errors::InvalidArgument(
        "Must specify at least one target to fetch or execute.");
  }

  std::unordered_set<string> endpoints;
  for (const auto& feed_rewrite : feed_rewrites) {
    auto result = endpoints.insert(feed_rewrite->endpoint_name()); //应该仔细看一下endpoint_name()这个东西，这个东西比较关键
    if (!result.second) {
      return errors::InvalidArgument("Endpoint \"",
                                     feed_rewrite->endpoint_name(),
                                     "\" fed more than once.");
    }
  }

  for (const auto& fetch_rewrite : fetch_rewrites) {
    if (endpoints.count(fetch_rewrite->endpoint_name()) > 0) {
      return errors::InvalidArgument(fetch_rewrite->endpoint_name(),
                                     " is both fed and fetched.");
    }
  } //判断feed_rewrite和fetch_rewrite是不是有重合的

  // A separate index mapping name to Node*, for use by FeedInputs,
  // FetchOutputs, and PruneForTargets
  NameIndex name_index;
  name_index.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    name_index[n->name()] = n; 
  } //将图中所有的node都传到这个字典中来

  // Add the feeds.  This may replace nodes in the graph, including the nodes
  // currently listed in "fetch_rewrites".  We pass "name_index" so the index is
  // kept up to date.
  if (!feed_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
        FeedInputs(g, feed_rewrites, &name_index, &out_metadata->feed_types)); 
  } //这个函数用来修改图

  // Add the fetch nodes, also updating "name_index".
  std::vector<Node*> fetch_nodes;
  if (!fetch_rewrites.empty()) {
    TF_RETURN_IF_ERROR(FetchOutputs(g, fetch_rewrites, &name_index,
                                    &fetch_nodes, &out_metadata->fetch_types)); //out_metadata->fetch_types应该对应的是所有节点的type
  }

  // Prune the graph to only compute what is needed for the fetch nodes and the
  // target nodes.
  if (!fetch_nodes.empty() || !target_node_names.empty()) {
    TF_RETURN_IF_ERROR(
        PruneForTargets(g, name_index, fetch_nodes, target_node_names)); //根据name_index、fetch_nodes、和target_node_names来对图进行剪枝
  }                                                                      //target_node_names这个时候并不知道有什么用

  return Status::OK();
}

```
可以看出，节点创建的关键点在于FeedInputs和FetchOutputs
```C++
Status FeedInputs(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    NameIndex* name_index, DataTypeVector* out_feed_types) {
  out_feed_types->clear();
  out_feed_types->reserve(feed_rewrites.size());
  for (size_t i = 0; i < feed_rewrites.size(); ++i) {
    const string& t = feed_rewrites[i]->endpoint_name();
    TensorId id(ParseTensorName(t));   //这个函数暂时不太清楚是怎么实现的，可能是关键点

    auto iter = name_index->find(id.first);    //在name_index中找到feed_rewrites中对应的node
    if (iter == name_index->end()) {
      return errors::NotFound("FeedInputs: unable to find feed output ", t);
    }
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);  
    if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument(
          "FeedInputs: ", t, " should have output index < ", n->num_outputs());
    } //看起来Tensorid 的first应该是name, second应该是输出节点数，这里就有一个疑问，n-> num_outputs()是如何定义的？

    Node* feed_node;
    TF_RETURN_IF_ERROR(
        feed_rewrites[i]->AddNode(g, {n, id.second}, &feed_node));  //{n,id.second}说明了该节点对应的输出节点, 那么node是哪里出来的？？？ 

    // Update name_index
    (*name_index)[feed_node->name()] = feed_node;  //更改节点之后图的节点列表也需要更新
    // Duplicate control edges aren't allowed, but feed_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(g->source_node(), feed_node, true); //为更改之后的节点添加控制边

    // Look through edges coming out of "n" for edges whose src_output() index
    // matches "output_index".  If found, replace the edges with a connection
    // from the special feed node.
    std::vector<const Edge*> to_remove;
    for (const Edge* e : n->out_edges()) {
      if (e->src_output() == id.second) {
        to_remove.emplace_back(e);
      } else if (e->src_output() == Graph::kControlSlot &&
                 (n->type_string() == "Placeholder" ||
                  n->type_string() == "PlaceholderV2")) {
        // When feeding a Placeholder node, any outgoing control edges
        // will be replaced with a control edge from the replacement
        // feed_node.
        // TODO(josh11b,mrry): Come up with a more elegant way of addressing
        // the general version of this problem.
        to_remove.emplace_back(e);
      }
    } //这里就有个问题 新生成的Arg的type_string应该是啥？ 这时候可能得看一些节点类的定义

    for (const Edge* e : to_remove) {
      if (e->src_output() == id.second) {
        g->AddEdge(feed_node, 0, e->dst(), e->dst_input());
      } else {
        CHECK_EQ(Graph::kControlSlot, e->src_output());
        // Duplicate control edges aren't allowed, but feed_node was *just*
        // created so there's no need to check for a duplicate.
        g->AddControlEdge(feed_node, e->dst(), true);
      }
      g->RemoveEdge(e);
    }
    out_feed_types->push_back(BaseType(n->output_type(id.second)));
  }
  return Status::OK();
}
```
在上边的代码中，关键问题是根据节点和节点的输出来添加ArgNode，那么这个时候关键问题就是ArgNode究竟是怎么来的,tensorflow写了一个类用来新建ArgNode。该类为ArgFeedRewrite，定义在tensorflow/core/graph/subgraph.h中。该类为PruneRewrite类的一个子类，并重写了初始化及AddNode函数。在初始化阶段定义了该类的成员变量arg_index以及父类的成员变量endpoint_name以及device_info。
```C++
class ArgFeedRewrite : public PruneRewrite {
 public:
  ArgFeedRewrite(const string* endpoint_name,
                 const DeviceAttributes* device_info, int32 arg_index)
      : PruneRewrite(endpoint_name, device_info), arg_index_(arg_index) {}
  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override;

 private:
  const int32 arg_index_;
};

```
下边的代码片段为在RewriteGraphForExecution中ArgFeddRewrite的调用方式。可以看出，feed_outputs就是对应这endpoint_name, i则是需要添加的Arg的index
```C++
  if (use_function_convention) {
    for (size_t i = 0; i < fed_outputs.size(); ++i) {
      feed_rewrites.emplace_back(new ArgFeedRewrite(
          &fed_outputs[i], &device_info, static_cast<int32>(i)));
    }
    
 feed_rewrites[i]->AddNode(g, {n, id.second}, &feed_node))  //这里又遇到了id.second是什么的问题了
   
```
然后是重点，该类的AddNode实现~这个函数简直是简单的让人抓狂。就只能继续看NodeBuilder的实现了。
```C++
Status ArgFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                               Node** out_node) {
  // NOTE(mrry): We must include the index as part of the node
  // name, because _Arg is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_arg_", feed_tensor.node->name(), "_",
                                  feed_tensor.index, "_", arg_index_),
                  "_Arg")
          .Attr("T", BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("index", arg_index_)
          .Finalize(g, out_node));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
} 
```
NodeBuiler的定义在/tensorflow/core/graph/node_builder.cc
```C++
NodeBuilder::NodeBuilder(StringPiece name, StringPiece op_name,
                         const OpRegistryInterface* op_registry)
    : def_builder_(name, op_name, op_registry) {}

NodeBuilder::NodeBuilder(StringPiece name, const OpDef* op_def)
    : def_builder_(name, op_def) {}
Status NodeBuilder::Finalize(Graph* graph, Node** created_node) const {
  // In case of error, set *created_node to nullptr.
  if (created_node != nullptr) *created_node = nullptr;
  if (!errors_.empty()) {
    return errors::InvalidArgument(str_util::Join(errors_, "\n"));
  }

  NodeDef node_def;
  TF_RETURN_IF_ERROR(def_builder_.Finalize(&node_def));   //先调用def_builder_中的Finalize
  TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, def_builder_.op_def())); 
  TF_RETURN_IF_ERROR(
      CheckOpDeprecation(def_builder_.op_def(), graph->versions().producer()));
  Status status;
  Node* node = graph->AddNode(node_def, &status);
  if (!status.ok()) return status;

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].node != nullptr) {  // Skip back edges.
      graph->AddEdge(inputs_[i].node, inputs_[i].index, node, i);
    }
  } //如果节点有输入的话，给输入添加边
  for (Node* control_input : control_inputs_) {
    graph->AddControlEdge(control_input, node);
  } //如果有控制边的话，就添加控制边
  if (created_node != nullptr) *created_node = node;
  return Status::OK();
}   

```
其中 def_builder_的类型是NodeDefBuilder, 定义在tensorflow/core/node_def_builder.h中
```C++
NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
                               const OpRegistryInterface* op_registry) {
  node_def_.set_name(string(name));
  const Status status = op_registry->LookUpOpDef(string(op_name), &op_def_); //这里从所有的op中根据名称找到相应的op并将op_def丢到op_def_中
  if (status.ok()) {
    Initialize(); 
  } else {
    errors_.push_back(status.error_message());
    inputs_specified_ = 0;
  }
}  
void NodeDefBuilder::Initialize() {
  inputs_specified_ = 0;
  node_def_.set_op(op_def_->name());  //set_op这个函数是干嘛用的 应该看一下
}

Status NodeDefBuilder::Finalize(NodeDef* node_def) const {
  const std::vector<string>* errors_ptr = &errors_;
  std::vector<string> errors_storage;
  if (op_def_ != nullptr && inputs_specified_ < op_def_->input_arg_size()) {
    // Since this is a const method, to add an error, we have to make
    // a copy of the existing errors.
    errors_storage = errors_;
    errors_storage.push_back(
        strings::StrCat(inputs_specified_, " inputs specified of ",
                        op_def_->input_arg_size(), " inputs in Op"));
    errors_ptr = &errors_storage;
  }

  if (!errors_ptr->empty()) {
    if (errors_ptr->size() == 1) {
      if (op_def_ == nullptr) {
        return errors::InvalidArgument((*errors_ptr)[0],
                                       " while building NodeDef '",
                                       node_def_.name(), "'");
      }
      return errors::InvalidArgument(
          (*errors_ptr)[0], " while building NodeDef '", node_def_.name(),
          "' using ", SummarizeOpDef(*op_def_));
    } else {
      return errors::InvalidArgument(
          errors_ptr->size(), " errors while building NodeDef '",
          node_def_.name(), "' using ", SummarizeOpDef(*op_def_), ":\n",
          str_util::Join(*errors_ptr, "\n"));
    }
  } else {
    NodeDef node_def_backup;
    if (node_def == nullptr) node_def = &node_def_backup;
    *node_def = node_def_;

    // Add control inputs after the regular inputs.
    for (const auto& control_input : control_inputs_) {
      node_def->add_input(strings::StrCat("^", control_input));
    }

    // Add default values for unspecified attrs.
    AddDefaultsToNodeDef(*op_def_, node_def);

    return Status::OK();
  }
}
```
