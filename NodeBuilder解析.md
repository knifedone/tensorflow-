```C++
Status DirectSession::CreateGraphs(
    const BuildGraphOptions& subgraph_options,
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    RunStateArgs* run_state_args, DataTypeVector* input_types,
    DataTypeVector* output_types, int64* collective_graph_key) {
  mutex_lock l(graph_state_lock_);
  std::unique_ptr<ClientGraph> client_graph;

  std::unique_ptr<GraphExecutionState> temp_exec_state_holder;
  GraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new GraphExecutionState for every new unseen graph,
    // and then place it.
    GraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForPrunedGraph(
        execution_state_->original_graph_def().library(), prune_options,
        execution_state_->original_graph_def(), subgraph_options,
        &temp_exec_state_holder, &client_graph));
    execution_state = temp_exec_state_holder.get();
  } else {
    execution_state = execution_state_.get();
    TF_RETURN_IF_ERROR(
        execution_state->BuildGraph(subgraph_options, &client_graph));
  }
  *collective_graph_key = client_graph->collective_graph_key;

  if (subgraph_options.callable_options.feed_size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.callable_options.feed_size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  if (subgraph_options.callable_options.fetch_size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.callable_options.fetch_size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  auto current_stateful_placements = execution_state->GetStatefulPlacements();
  // Update our current state based on the execution_state's
  // placements.  If there are any mismatches for a node,
  // we should fail, as this should never happen.
  for (auto placement_pair : current_stateful_placements) {
    const string& node_name = placement_pair.first;
    const string& placement = placement_pair.second;
    auto iter = stateful_placements_.find(node_name);
    if (iter == stateful_placements_.end()) {
      stateful_placements_.insert(std::make_pair(node_name, placement));
    } else if (iter->second != placement) {
      return errors::Internal(
          "Stateful placement mismatch. "
          "Current assignment of ",
          node_name, " to ", iter->second, " does not match ", placement);
    }
  }

  stateful_placements_ = execution_state->GetStatefulPlacements();

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
  }

  // Partition the graph across devices.
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.flib_def = &client_graph->graph.flib_def();
  popts.control_flow_added = false;

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }

  for (const auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(
        new Graph(client_graph->flib_def.get()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second,
                                              device_graph.get()));
    outputs->emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

    VLOG(2) << "Created " << DebugString(graph->get()) << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    s = d->MaybeRewriteGraph(graph);
    if (!s.ok()) {
      break;
    }
  }
  *flib_def = std::move(client_graph->flib_def);
  std::swap(*input_types, client_graph->feed_types);
  std::swap(*output_types, client_graph->fetch_types);
  return s;
}
```

```C++
Status GraphExecutionState::BuildGraph(const BuildGraphOptions& options,
                                       std::unique_ptr<ClientGraph>* out) {
  VLOG(1) << "BuildGraph";
  if (!graph_) {
    // It is only valid to call this method directly when the original graph
    // was created with the option `place_pruned_graph == false`.
    return errors::Internal(
        "Attempted to prune a graph that has not been fully initialized.");
  }

  // Grappler optimization might change the structure of a graph itself, and
  // also it can add/prune functions to/from the library.
  std::unique_ptr<Graph> optimized_graph;
  std::unique_ptr<FunctionLibraryDefinition> optimized_flib;

  Status s = OptimizeGraph(options, &optimized_graph, &optimized_flib);
  if (!s.ok()) {
    VLOG(2) << "Grappler optimization failed. Error: " << s.error_message();
    // Simply copy the original graph and the function library if we couldn't
    // optimize it.
    optimized_graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*graph_, optimized_graph.get());
    optimized_flib.reset(new FunctionLibraryDefinition(*flib_def_));
  }

  subgraph::RewriteGraphMetadata rewrite_metadata;
  if (session_options_ == nullptr ||
      !session_options_->config.graph_options().place_pruned_graph()) {
    TF_RETURN_IF_ERROR(
        PruneGraph(options, optimized_graph.get(), &rewrite_metadata));
  } else {
    // This GraphExecutionState represents a graph that was
    // pruned when this was constructed, so we copy the metadata from
    // a member variable.
    CHECK(rewrite_metadata_);
    rewrite_metadata = *rewrite_metadata_;
  }

  CHECK_EQ(options.callable_options.feed_size(),
           rewrite_metadata.feed_types.size());
  CHECK_EQ(options.callable_options.fetch_size(),
           rewrite_metadata.fetch_types.size());

  // TODO(andydavis): Clarify optimization pass requirements around CostModel.
  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &optimized_graph;
  optimization_options.flib_def = optimized_flib.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));

  int64 collective_graph_key = options.collective_graph_key;
  if (collective_graph_key == BuildGraphOptions::kNoCollectiveGraphKey) {
    // BuildGraphOptions does not specify a collective_graph_key.  Check all
    // nodes in the Graph and FunctionLibraryDefinition for collective ops and
    // if found, initialize a collective_graph_key as a hash of the ordered set
    // of instance keys.
    std::set<int32> instance_key_set;
    for (Node* node : optimized_graph->nodes()) {
      if (node->IsCollective()) {
        int32 instance_key;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(node->attrs(), "instance_key", &instance_key));
        instance_key_set.emplace(instance_key);
      } else {
        const FunctionDef* fdef = optimized_flib->Find(node->def().op());
        if (fdef != nullptr) {
          for (const NodeDef& ndef : fdef->node_def()) {
            if (ndef.op() == "CollectiveReduce" ||
                ndef.op() == "CollectiveBcastSend" ||
                ndef.op() == "CollectiveBcastRecv") {
              int32 instance_key;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "instance_key", &instance_key));
              instance_key_set.emplace(instance_key);
            }
          }
        }
      }
    }
    if (!instance_key_set.empty()) {
      uint64 hash = 0x8774aa605c729c72ULL;
      for (int32 instance_key : instance_key_set) {
        hash = Hash64Combine(instance_key, hash);
      }
      collective_graph_key = hash;
    }
  }

  // Copy the extracted graph in order to make its node ids dense,
  // since the local CostModel used to record its stats is sized by
  // the largest node id.
  std::unique_ptr<ClientGraph> dense_copy(
      new ClientGraph(std::move(optimized_flib), rewrite_metadata.feed_types,
                      rewrite_metadata.fetch_types, collective_graph_key));
  CopyGraph(*optimized_graph, &dense_copy->graph);

  // TODO(vrv): We should check invariants of the graph here.

  *out = std::move(dense_copy);
  return Status::OK();
}
```
```C++
Status GraphExecutionState::PruneGraph(
    const BuildGraphOptions& options, Graph* graph,
    subgraph::RewriteGraphMetadata* out_rewrite_metadata) {
  std::vector<std::unique_ptr<subgraph::PruneRewrite>> feed_rewrites;
  feed_rewrites.reserve(options.callable_options.feed_size());
  std::vector<std::unique_ptr<subgraph::PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(options.callable_options.fetch_size());
  if (options.use_function_convention) {
    std::vector<TensorAndDevice> tensors_and_devices;
    for (int i = 0; i < options.callable_options.feed_size(); ++i) {
      // WARNING: feed MUST be a reference, since ArgFeedRewrite and
      // tensors_and_devices holds on to its address.
      const string& feed = options.callable_options.feed(i);
      const DeviceAttributes* device_info;
      TF_RETURN_IF_ERROR(LookupDevice(*device_set_, feed,
                                      options.callable_options.feed_devices(),
                                      &device_info));
      feed_rewrites.emplace_back(
          new subgraph::ArgFeedRewrite(&feed, device_info, i));  //假设feed_dict{'a':1,'b':2},那么b对应的index为1
      tensors_and_devices.push_back({ParseTensorName(feed), device_info});
    }
    if (!options.callable_options.fetch_devices().empty() &&
        !options.callable_options.fetch_skip_sync()) {
      return errors::Unimplemented(
          "CallableOptions.fetch_skip_sync = false is not yet implemented. You "
          "can set it to true instead, but MUST ensure that Device::Sync() is "
          "invoked on the Device corresponding to the fetched tensor before "
          "dereferencing the Tensor's memory.");
    }
    for (int i = 0; i < options.callable_options.fetch_size(); ++i) {
      // WARNING: fetch MUST be a reference, since RetvalFetchRewrite and
      // tensors_and_devices holds on to its address.
      const string& fetch = options.callable_options.fetch(i);
      const DeviceAttributes* device_info;
      TF_RETURN_IF_ERROR(LookupDevice(*device_set_, fetch,
                                      options.callable_options.fetch_devices(),
                                      &device_info));
      fetch_rewrites.emplace_back(
          new subgraph::RetvalFetchRewrite(&fetch, device_info, i));
      tensors_and_devices.push_back({ParseTensorName(fetch), device_info});
    }
    TF_RETURN_IF_ERROR(
        ValidateFeedAndFetchDevices(*graph, tensors_and_devices));
  } else {
    if (!options.callable_options.feed_devices().empty() ||
        !options.callable_options.fetch_devices().empty()) {
      return errors::Unimplemented(
          "CallableOptions::feed_devices and CallableOptions::fetch_devices "
          "to configure feeding/fetching tensors to/from device memory is not "
          "yet supported when using a remote session.");
    }
    const DeviceAttributes* device_info =
        &device_set_->client_device()->attributes();
    for (const string& feed : options.callable_options.feed()) {
      feed_rewrites.emplace_back(
          new subgraph::RecvFeedRewrite(&feed, device_info));
    }
    for (const string& fetch : options.callable_options.fetch()) {
      fetch_rewrites.emplace_back(
          new subgraph::SendFetchRewrite(&fetch, device_info));
    }
  }

  for (const TensorConnection& tensor_connection :
       options.callable_options.tensor_connection()) {
    Node* from_node = nullptr;
    TensorId from_id(ParseTensorName(tensor_connection.from_tensor()));

    for (Node* n : graph->nodes()) {
      if (n->name() == from_id.first) {
        from_node = n;
        break;
      }
    }
    if (from_node == nullptr) {
      return errors::InvalidArgument(
          "Requested tensor connection from unknown node: \"",
          tensor_connection.to_tensor(), "\".");
    }
    if (from_id.second >= from_node->num_outputs()) {
      return errors::InvalidArgument(
          "Requested tensor connection from unknown edge: \"",
          tensor_connection.to_tensor(),
          "\" (actual number of outputs = ", from_node->num_outputs(), ").");
    }

    feed_rewrites.emplace_back(new TensorConnectionPruneRewrite(
        &tensor_connection.to_tensor(), {from_node, from_id.second}));
  }

  std::vector<string> target_node_names(
      options.callable_options.target().begin(),
      options.callable_options.target().end());
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph, feed_rewrites, fetch_rewrites, target_node_names,
      out_rewrite_metadata));

  CHECK_EQ(out_rewrite_metadata->feed_types.size(),
           options.callable_options.feed_size() +
               options.callable_options.tensor_connection_size());
  for (int i = 0; i < options.callable_options.tensor_connection_size(); ++i) {
    out_rewrite_metadata->feed_types.pop_back();
  }
  return Status::OK();
}
```

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
TensorId ParseTensorName(StringPiece name) {
  // Parse either a name, ^name, or name:digits.  To do so, we go backwards from
  // the end of the string, skipping over a run of digits.  If we hit a ':'
  // character, then we know we are in the 'name:digits' regime.  Otherwise, we
  // see if the name starts with '^', indicating a control edge. If we find
  // neither ':' nor '^' characters, the output index is implicitly 0, and the
  // whole name string forms the first part of the tensor name.
  const char* base = name.data();
  const char* p = base + name.size() - 1;
  unsigned int index = 0;
  unsigned int mul = 1;
  while (p > base && (*p >= '0' && *p <= '9')) {
    index += ((*p - '0') * mul);
    mul *= 10;
    p--;
  }
  TensorId id;
  if (p > base && *p == ':' && mul > 1) {
    id.first = StringPiece(base, p - base);
    id.second = index;
  } else if (str_util::StartsWith(name, "^")) {
    // Control edge
    id.first = StringPiece(base + 1);
    id.second = Graph::kControlSlot;
  } else {
    id.first = name;
    id.second = 0;
  }
  return id;
}
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
  TF_RETURN_IF_ERROR(def_builder_.Finalize(&node_def));   //先调用def_builder_中的Finalize 将node_def中的默认值设置好
  TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, def_builder_.op_def())); 
  TF_RETURN_IF_ERROR(
      CheckOpDeprecation(def_builder_.op_def(), graph->versions().producer()));
  Status status;
  Node* node = graph->AddNode(node_def, &status); //根据node_def来添加节点
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
其中 def_builder_的类型是NodeDefBuilder, 定义在tensorflow/core/framework/node_def_builder.h中
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
    AddDefaultsToNodeDef(*op_def_, node_def); //根据op_def来为node_def设置初始值

    return Status::OK();
  }
}
```
```C++
Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def) {
  if (node_def.op() != op_def.name()) {
    return errors::InvalidArgument("NodeDef op '", node_def.op(),
                                   "' does not match ", SummarizeOpDef(op_def),
                                   "; NodeDef: ", SummarizeNodeDef(node_def));
  }

  bool seen_control = false;
  size_t num_inputs = 0;
  // TODO(josh11b): Unify the input field validation.
  for (const string& input : node_def.input()) {
    if (str_util::StartsWith(input, "^")) {
      seen_control = true;
      if (input.find(':') != string::npos) {
        return errors::InvalidArgument(
            "Control input '", input,
            "' must not have ':' in NodeDef: ", SummarizeNodeDef(node_def));
      }
    } else if (seen_control) {
      return errors::InvalidArgument(
          "Non-control input '", input,
          "' after control input in NodeDef: ", SummarizeNodeDef(node_def));
    } else {
      ++num_inputs;
    }
  } //得到这个node的input个数，并判断该node是否有控制边， 控制边只能有一条

  std::unordered_map<string, const OpDef::AttrDef*> op_attrs;
  for (const auto& attr : op_def.attr()) {
    if (!gtl::InsertIfNotPresent(&op_attrs, attr.name(), &attr)) {
      return errors::InvalidArgument("OpDef has duplicate attr name '",
                                     attr.name(),
                                     "': ", SummarizeOpDef(op_def));
    }
  } //新建一个attr_name:attr的map
  for (const auto& attr : node_def.attr()) {
    // Allow internal optional attributes with names starting with "_".
    if (str_util::StartsWith(attr.first, "_")) {
      continue;
    }
    auto iter = op_attrs.find(attr.first);
    if (iter == op_attrs.end()) {
      // A common cause of this error is that TensorFlow has made a
      // backwards-compatible change to the NodeDef (e.g., adding a
      // new attr with a default value), but the binary consuming the
      // NodeDef does not know about the new attribute; the solution
      // in these cases is to ensure that the binary consuming the
      // NodeDef is built with a version of TensorFlow no earlier than
      // the binary producing it.
      return errors::InvalidArgument(
          "NodeDef mentions attr '", attr.first, "' not in ",
          SummarizeOpDef(op_def), "; NodeDef: ", SummarizeNodeDef(node_def),
          ". (Check whether your GraphDef-interpreting binary is up to date "
          "with your GraphDef-generating binary.).");
    }
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ValidateAttrValue(attr.second, *iter->second),
        "; NodeDef: ", SummarizeNodeDef(node_def), "; ",
        SummarizeOpDef(op_def));
    // Keep track of which attr names have (not) been found in the NodeDef.
    op_attrs.erase(iter);
  }

  // Were all attrs in the OpDef found in the NodeDef?
  if (!op_attrs.empty()) {
    string attrs;
    for (const auto& attr_pair : op_attrs) {
      if (!attrs.empty()) strings::StrAppend(&attrs, "', '");
      strings::StrAppend(&attrs, attr_pair.first);
    }
    return errors::InvalidArgument("NodeDef missing attr",
                                   op_attrs.size() == 1 ? " '" : "s '", attrs,
                                   "' from ", SummarizeOpDef(op_def),
                                   "; NodeDef: ", SummarizeNodeDef(node_def));
  }

  // Validate the number of inputs.
  DataTypeVector inputs, outputs;
  TF_RETURN_IF_ERROR(InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  if (num_inputs != inputs.size()) {
    return errors::InvalidArgument(
        "NodeDef expected inputs '", DataTypeVectorString(inputs),
        "' do not match ", num_inputs, " inputs specified; ",
        SummarizeOpDef(op_def), "; NodeDef: ", SummarizeNodeDef(node_def));
  }

  return Status::OK();
}
```
```C++
void Graph::RemoveNode(Node* node) {
  TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());

  // Remove any edges involving this node.
  while (!node->in_edges_.empty()) {
    RemoveEdge(*node->in_edges_.begin());
  }
  while (!node->out_edges_.empty()) {
    RemoveEdge(*node->out_edges_.begin());
  }
  ReleaseNode(node);
}


```
