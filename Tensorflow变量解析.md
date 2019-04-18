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
    self._shape = initial_value.shape
    self._unique_id = unique_id
    self._initial_value = initial_value if self._in_graph_mode else None
    self._handle_name = handle_name + ":0"
    self._dtype = initial_value.dtype.base_dtype
    self._constraint = constraint
    
        
```
