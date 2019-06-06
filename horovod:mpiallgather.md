```C++
MPIAllgather::MPIAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mpi_context_(mpi_context) {}

bool MPIAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status MPIAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  auto* recvcounts = new int[global_state_->size]();
  auto* displcmnts = new int[global_state_->size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_state_->size]();
    entry_component_offsets[ec] = new int64_t[global_state_->size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);  //tensor_states_[entries.name]设成activate
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }   //set e.context->AllocateOutput(outputshape,&e.output) for e in entries
  timeline.ActivityEndAll(entries); 

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);
    
  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  const void* sendbuf = nullptr;
  void* buffer_data;
  int64_t total_num_elements = NumElements(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data); \\这个element_size是什么？  
    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  auto dtype = mpi_context_->GetMPIDataType(first_entry.tensor->dtype());
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE,
                          (int) total_num_elements,
                          dtype,
                          buffer_data,
                          recvcounts,
                          displcmnts,
                          dtype,
                          mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityE
  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}
```
```C++
Status AllgatherOp::AllocateOutput(std::vector<TensorTableEntry>& entries, const Response& response,
                                   int64_t**& entry_component_sizes, int*& recvcounts) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    int64_t total_entry_dimension_size = 0;
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_state_->size; ++rc) {
      auto component_size = tensor_sizes[ec * global_state_->size + rc];
      total_entry_dimension_size += component_size;
      recvcounts[rc] += component_size * single_slice_shape.num_elements();
      entry_component_sizes[ec][rc] =
          component_size * single_slice_shape.num_elements();
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t) total_entry_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    Status status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      return status;
    }
  }

  return Status::OK();
}

```
```C++
int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE,
                          (int) total_num_elements,
                          dtype,
                          buffer_data,
                          recvcounts,
                          displcmnts,
                          dtype,
                          mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
```
