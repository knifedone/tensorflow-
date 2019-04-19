Tensorflow中有许多不同的变量类型，在这个blog中将会对tensorflow/python/ops/variables.py进行解析。

RefVariable()
----
RefVariable()类继承自 Variable_v1,是目前版本tensorflow的主要变量类型(r1.13)
该类主要由以下属性所构成,同时，该类还重载了许多Tensor中应有的功能,例如assign等。该类所支持的功能较多，详情见https://www.tensorflow.org/api_docs/python/tf/Variable
```python
self._in_graph_mode = True #默认为True
self._graph_key = ops.get_default_graph()._graph_key #记录这个默认图的key，以便后续恢复图
self.update_uid  #
self.trainable #该变量是否可以训练
self._initial_value #一个tensor，记录这个Variable的value值
self._variable #返回一个op,这个op的内容是——initial_value
self._initializer_op #一个初始化op
self._caching_device #
self._save_slice_info = None
self._constraint #
self._variable #记录一些variable中的reference,是一个tensor类型。
```

partitionedVariable()
----

```python
class PartitionedVariable(object):
  def __init__(self, name, shape, dtype, variable_list, partitions):
    self._variable_list = sorted(
        variable_list, key=lambda v: v._get_save_slice_info().var_offset)
    self._name = name
    self._shape = shape
    self._dtype = dtype
    self._partitions = partitions
    self._as_tensor = None
  def __iter__(self):
    """Return an iterable for accessing the underlying partition Variables."""
    return iter(self._variable_list)
  def __len__(self):
    num_partition_axes = len(self._partition_axes())
    if num_partition_axes > 1:
      raise ValueError("Cannot get a length for %d > 1 partition axes"
                       % num_partition_axes)
    return len(self._variable_list)
  def _partition_axes(self):
    if all(p == 1 for p in self._partitions):
      return [0]
    else:
      return [i for i, p in enumerate(self._partitions) if p > 1]
  def _concat(self): #将所有的Tensor聚合成一个tensor
    if len(self._variable_list) == 1:
      with ops.name_scope(None):
        return array_ops.identity(self._variable_list[0], name=self._name)

    partition_axes = self._partition_axes()

    if len(partition_axes) > 1:
      raise NotImplementedError(
          "Cannot concatenate along more than one dimension: %s.  "
          "Multi-axis partition concat is not supported" % str(partition_axes))
    partition_ix = partition_axes[0]

    with ops.name_scope(self._name + "/ConcatPartitions/"):
      concatenated = array_ops.concat(self._variable_list, partition_ix)

    with ops.name_scope(None):
      return array_ops.identity(concatenated, name=self._name)
  

```
ResourceVariable
----
ResourceVariable可以被看做是RefVariable的加强版。简单的说，ResourceVariable可以在一系列的调用过程中保持状态不变，更加安全。与RefVariable类似，该类同样继承自Variable_v1，有相似的特性，同样可以被用来作为图中其他op的输入，并且具有Tensor中的特性。
与RefVariable不同的是，ResourceVariable有更加严谨的定义。ResourceVariable的每一次使用都会在图中生成一个read_value的op.这些read_value op返回的tensor可以保证 1 任何与read_value有依赖的对ResourceVariable的更改都可以被追溯。 2 保证任何与read_value有依赖的操作都不会被更改。
```python
class ResourceVariable(variables.VariableV1):
  def __init__(self,
             initial_value=None,
             trainable=True,
             collections=None,
             validate_shape=True,  # pylint: disable=unused-argument
             caching_device=None,
             name=None,
             dtype=None,
             variable_def=None,
             import_scope=None,
             constraint=None,
             distribute_strategy=None,
             synchronization=None,
             aggregation=None):
  _init_from_proto or _init_from_args
```
事实上，ResourceVariable和RefVariable的初始化暴露的接口是完全相同的，故在此不过多介绍。
在init_from_args这个函数中，与RefVariable相比，增加了对Eager_mode的支持。在Ref的定义中，in_graph被默认为True,但在ResourceVariable中，这个属性的值取决于ResourceVariable是否处于Eager_mode。
```python
  def _init_from_args(self,
                      initial_value=None,
                      trainable=True,
                      collections=None,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      constraint=None,
                      synchronization=None,
                      aggregation=None):
    self._update_uid
    self._synchronization = synchronization
    self._aggregation = aggregation
    self._trainable = trainable
    self._save_slice_info = None
    self._graph_key = ops.get_default_graph()._graph_key
    self._handle = eager_safe_variable_handle(
              initial_value=initial_value,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode)
              #这里生成了一个handle(其实就是一个tensor),这个handle保存了这个Variable的一些信息，例如shape, type, container等。
              #据官方文档所说，在这个Variable的后续op中，都将调用这个handle
    self._shape = initial_value.shape
    self._unique_id = unique_id
    self._initial_value = initial_value if self._in_graph_mode else None
    self._handle_name = handle_name + ":0"
    self._dtype = initial_value.dtype.base_dtype
    self._constraint = constraint
    
        
```
```python
def eager_safe_variable_handle(initial_value, shared_name, name, graph_mode):
  """Creates a variable handle with information to do shape inference.

  The shape and dtype are read from `initial_value` and stored in the returned
  resource tensor's handle data.

  If `initial_value.dtype == tf.variant`, we additionally extract the handle
  data (if any) from `initial_value` and append it to the `handle_data`.
  In this case, the returned tensor's handle data is in the form

  
  is_set: true
  shape_and_type {
    shape {
      // initial_value.shape
    }
    dtype: DT_VARIANT
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[0]
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[1]
  }
  


  Ops that read from this tensor, such as `ReadVariableOp` and
  `AssignVariableOp`, know that `handle_data(handle).shape_and_type[1:]`
  correspond to the handle data of the variant(s) stored in the Variable.

  Args:
    initial_value: A `Tensor`.
    shared_name: A string.
    name: A string.
    graph_mode: A python bool.

  Returns:
    The handle, a `Tensor` of type `resource`.
  """
  shape = initial_value.get_shape()
  dtype = initial_value.dtype.base_dtype
  container = ops.get_default_graph()._container  # pylint: disable=protected-access
  if container is None:
    container = ""
  handle = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                                                   shared_name=shared_name,
                                                   name=name,
                                                   container=container)  
  #这里创建了一个handle用来处理ResourceVariable
  if graph_mode:
    full_handle_data = _combine_handle_data(handle, initial_value)
    _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
    return handle
  else:
    # We do not want two distinct ResourceVariable objects for the same
    # underlying resource in the runtime.
    # When in eager mode, explicitly ensure so here. When in graph mode, it's
    # ensured by always generating different variable names.
    exists = gen_resource_variable_ops.var_is_initialized_op(handle)
    if exists:
      raise ValueError("variable object with name '%s' already created. Use "
                       "get_variable() if reuse is desired." %
                       shared_name)
    with context.graph_mode(), ops.Graph().as_default() as graph:
      h = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                                                  shared_name=shared_name,
                                                  name=name,
                                                  container=container)

      # Tensor._handle_data contains information for the shape-inference code to
      # know the shape and dtype of the variable pointed to by a handle. Since
      # shape inference doesn't run in eager mode we copy this data here for
      # when the handle is captured by an eager mode function.
      # pylint: disable=protected-access
      full_handle_data = _combine_handle_data(h, initial_value)
      _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
      # pylint: enable=protected-access
    # Clean up op->graph->op reference cycles.
    ops.dismantle_graph(graph)
    return handle

```
ResourceVariable中新增加了一些read相关的函数，这些函数通常都会调用_handle接口，并向计算图中登记。
```python
  def _read_variable_op(self):
  #返回一个op
    variable_accessed(self)
    result = gen_resource_variable_ops.read_variable_op(self._handle,
                                                        self._dtype) #read_variable_op()生成了一个op,这个op的功能是从_handle中读取variable
    _maybe_set_handle_data(self._dtype, self._handle, result)  #暂时不知道这个是干嘛的....到时候先照着写吧。

    if not context.executing_eagerly():
      # Note that if a control flow context is active the input of the read op
      # might not actually be the handle. This line bypasses it.
      tape.record_operation(
          "ReadVariableOp", [result], [self._handle], lambda x: [x]) #把这个op添加到tape流中，官方注释说这样可以保证input不仅仅是handle
    return result
```

```python

```
