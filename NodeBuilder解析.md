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
