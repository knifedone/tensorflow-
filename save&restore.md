# Save和restore
这篇blog记录了一下train.Saver类的用法
## SaverDef
----
首先，介绍一下SaverDef消息机制，SaverDef是一种消息类型，承载了Saver中的各种设定。
```C
message SaverDef {
  // The name of the tensor in which to specify the filename when saving or
  // restoring a model checkpoint.
  string filename_tensor_name = 1;

  // The operation to run when saving a model checkpoint.
  string save_tensor_name = 2;

  // The operation to run when restoring a model checkpoint.
  string restore_op_name = 3;

  // Maximum number of checkpoints to keep.  If 0, no checkpoints are deleted.
  int32 max_to_keep = 4;

  // Shard the save files, one per device that has Variable nodes.
  bool sharded = 5;

  // How often to keep an additional checkpoint. If not specified, only the last
  // "max_to_keep" checkpoints are kept; if specified, in addition to keeping
  // the last "max_to_keep" checkpoints, an additional checkpoint will be kept
  // for every n hours of training.
  float keep_checkpoint_every_n_hours = 6;

  // A version number that identifies a different on-disk checkpoint format.
  // Usually, each subclass of BaseSaverBuilder works with a particular
  // version/format.  However, it is possible that the same builder may be
  // upgraded to support a newer checkpoint format in the future.
  enum CheckpointFormatVersion {
    // Internal legacy format.
    LEGACY = 0;
    // Deprecated format: tf.Saver() which works with tensorflow::table::Table.
    V1 = 1;
    // Current format: more efficient.
    V2 = 2;
  }
  CheckpointFormatVersion version = 7;
}
```
## Saver类
-----
Saver类是tensorflow存储及恢复模型的核心类
### Saver类的定义
```python
class Saver(object):
  def __init__(self,
               var_list=None,
               ...
               ):
    if not defer_build:
      self.build() #如果之前没有build的话，就build()
               
```
这里调用了一个build()函数，其中涉及到了BulkSaverBuilder()和variables()这两个类，关于这两个类的说明一会儿会说到
```python
def build(self,checkpoint_path, build_save, build_restore):
  if not self.saver_def:
    if _builder is None:
      self._builder = BulkSaverBuilder(self._write_version) #如果之前没有生成过build就调用BulkSaverBuilder()这个类来生成builder
    if self._var_list is None:
      self._var_list = variables._all_saveable_objects()  #获取可以保存的变量列表
    self.saver_def = self._builder._build_internal(...) #调用_build_internal来生成一个SaverDef消息
  elif self.saver_def and self._name: 
      #如果在之前已经生成过saver_def变量则直接在原有的Saver_def的基础上检查和添加,不重要，不写了
      pass
```
build_internal()这个函数则生成了一个默认图，并在图中添加了filename,saveop,restoreop三个结点：
```python
def _build_internal(self,
                      names_to_saveables,
                      reshape=False,
                      sharded=False,
                      max_to_keep=5,
                      keep_checkpoint_every_n_hours=10000.0,
                      name=None,
                      restore_sequentially=False,
                      filename="model",
                      build_save=True,
                      build_restore=True):
  #names_to_saveables是需要保存的变量的字典，其中key代表名字，value可能是Variable或者是SaveableObject.
  #字典中的每个变量都对应这checkpoint中的相应变量
    saveables = saveable_object_util.validate_and_slice_inputs(
        names_to_saveables) #返回list类型的saveable变量
    with ops.name_scope(name, "save",
                      [saveable.op for saveable in saveables]) as name:
      # Add a placeholder string tensor for the filename.
      filename_tensor = array_ops.placeholder_with_default(
          filename or "model", shape=(), name="filename")
      # Keep the name "Const" for backwards compatibility.
      filename_tensor = array_ops.placeholder_with_default(
          filename_tensor, shape=(), name="Const")
      # Add the save ops.
      if sharded:
        per_device = self._GroupByDevices(saveables)
        if build_save:
          save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
        if build_restore:
          restore_op = self._AddShardedRestoreOps(filename_tensor, per_device,
                                                  restore_sequentially, reshape)
      else:
        if build_save:
          save_tensor = self._AddSaveOps(filename_tensor, saveables)
        if build_restore:
          restore_op = self._AddRestoreOps(filename_tensor, saveables,
                                           restore_sequentially, reshape)
      #这里创建了filename_tensor,save_tensor以及restore_op节点
```
build_internal(）这个函数中调用了_AddSaveOps()等函数，这个函数是BulkSaverBuilder()类中的内容，故简单的对该类进行说明
#### BuldSaverBuilder()
初始化：
```python
class BaseSaverBuilder(object):

  def __init__(self, write_version=saver_pb2.SaverDef.V2):
    self._write_version = write_version
    
```
save_op()函数，该函数创建了一个op来保存saveables,该函数将saveables中的内容分离出来，并通过io_ops.save_v2生成了一个op,这个op的功能是将saveables中的内容写入filename_tensor所指向的文件中去
```python
  def save_op(self, filename_tensor, saveables):
    
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in saveables:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)
    return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices,
                            tensors) 
```
bulf_restore函数，该函数对每个saveable变量分别调用restore_op函数，并返回一个list,list中包含一系列从filename中读取saveable的op
```python
  def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                   restore_sequentially):
   
    del restore_sequentially
    all_tensors = []
    for saveable in saveables:
      if saveable.device:
        device = saveable_object_util.set_cpu0(saveable.device)
      else:
        device = None
      with ops.device(device):
        all_tensors.extend(
            self.restore_op(filename_tensor, saveable, preferred_shard))
    return all_tensors
```
restore_op：返回tensors,tensors中的每个元素都是一个从filename指向saveable的op
```python
  # pylint: disable=unused-argument
  def restore_op(self, filename_tensor, saveable, preferred_shard):
    
    tensors = []
    for spec in saveable.specs:
      tensors.append(
          io_ops.restore_v2(
              filename_tensor,
              [spec.name],
              [spec.slice_spec],
              [spec.dtype])[0])

    return tensors
 ```
AddSaveOps这个函数输出了一个从filename_tensor的op，该op存在单向依赖
```python
  def _AddSaveOps(self, filename_tensor, saveables):
    save = self.save_op(filename_tensor, saveables) #创建一个op，这个op是从filename_tensor中读取saveables
    return control_flow_ops.with_dependencies([save], filename_tensor)  #返回一个op 这个op在save之后才会被执行
  #相当于save这个op执行之后才会执行filename_tensor
  
  """
  以下是tensorflow官方文档对control_flow_ops.with_dependencies()这个函数的声明
  with_dependencies(dependencies, output_tensor, name=None)
  Produces the content of `output_tensor` only after `dependencies`.也就是说，该函数建立了一个从dependencies到output_tensor的依赖，只有依赖op全部完成，才会进行output_tensor
  """
_AddSjardedSaveOpsForV2这个函数与_AddSaveOps类似，是加了分块的版本
```python

  def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
    #不会详细展开
```
AddRestoreOps()这个函数返回一个op的聚合。 首先调用all_tensors得到一个tensor,tensor中包含从filename_tensor到saveable的op。为了提升效率并保证这些op有同时完成的依赖关系，将这些op聚合成一个op
```python
  def _AddRestoreOps(self,
                     filename_tensor,
                     saveables,
                     restore_sequentially,
                     reshape,
                     preferred_shard=-1,
                     name="restore_all"):
    """Add operations to restore saveables.

  
    all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard,
                                    restore_sequentially) #这里得到了一个tensor,tensor中的内容是从filename_tensor到saveable的op
    assign_ops = []
    idx = 0
    
    for saveable in saveables:
      shapes = None
      if reshape:
        # Compute the shapes, let the restore op decide if and how to do
        # the reshape.
        shapes = []
        for spec in saveable.specs:
          v = spec.tensor
          shape = v.get_shape()
          if not shape.is_fully_defined():
            shape = array_ops.shape(v)
          shapes.append(shape)
      saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
      idx += len(saveable.specs)
      assign_ops.append(saveable.restore(saveable_tensors, shapes)) #将tensor变成一个iterator op? #将tensor形式的op变成一个iter形式的op

    # Create a Noop that has control dependencies from all the updates.
    return control_flow_ops.group(*assign_ops, name=name) #把所有的restore 的op聚合在一起
```
### as_saver_def()
调用该函数可以生成一个SaverDer
```python
  def as_saver_def(self):
    return self.saver_def
```
### to_proto(export_scope=None)
该函数功能和as_saver_def()类似，只不过多了一个export_scope，只输出指定export_scope下的内容
```python
  def to_proto(export_scope=None):
    pass
    return saver_def
```
### from_proto(saver_def, import_scope=None）
顾名思义，在存在一个saverDef的protocol消息栈时，可以直接生成一个Saver类
```python
  def from_proto(saver_def, import_scope=None):
    return Saver(saver_def=saver_def, name=import_scope)
```
### last_checkpoints(self):
返回checkpoint列表
```python
@property
  def last_checkpoints(self):
    return list(self._CheckpointFilename(p) for p in self._last_checkpoints)
```
### save():
save函数接受以下输入：
*sess 用来存储变量的session
*save_path: 生成checkpoint文件名的前缀
*global_step
*lastest_filename 一个自动维护的过去保存的checkpoint列表
*meta_graph_suffix metaFraphDef文件的后缀
*write_meta_graph 一个boolen变量，决定是否记录checkpointStataProto
*strip_default_attrs: 一个boolen变量，如果为True,则默认值的状态将会被移除，该变量的存在可以增加graph的稳健性
*save_debug_info: 一个boolen变量，决定是否记录debug信息
save函数的输出为checkpoint的路径
```python
def save(self,...):
  checkpoint_file = save_path
  save_path_parent = os.path.dirname(save_path)
  model_checkpoint_path = sess.run(
              self.saver_def.save_tensor_name,
              {self.saver_def.filename_tensor_name: checkpoint_file}) #这个节点在_build_internal()中定义此处run了一下并返回了checkpoint的路径
  if write_state:
    self._RecordLastCheckpoint(model_checkpoint_path)
    checkpoint_management.update_checkpoint_state_internal(...)
    self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
  if write_meta_graph:
    meta_graph_filename = checkpoint_management.meta_graph_filename(
          checkpoint_file, meta_graph_suffix=meta_graph_suffix)
    with sess.graph.as_default():
      self.export_meta_graph(
          meta_graph_filename, strip_default_attrs=strip_default_attrs,
          save_debug_info=save_debug_info)
   return model_checkpoint_path
```

### restore(self,sess,save_path):
从save_path中将值赋到sess中
```python
def restore(self, sess, save_path):
  sess.run(self.saver_def.restore_op_name,
                 {self.saver_def.filename_tensor_name: save_path}) #从save_path中读取值并赋到restore_op_name中
  or
  self._object_restore_saver.restore(sess=sess, save_path=save_path) #从object_based中读取
   
```

### import_meta_graph(meta_graph_or_file,clear_devices,import_scope,##kwards):
从MetaGraphDef文件中重新生成一个graph。这中恢复方式和restore有所不同。restore是在已经有计算图的各个节点的情况下，通过restore()将其中的数值恢复，import_meta_graph()则是重新生成一张图。这种方式更利于模型的持久化保存。
```python
def import_meta_graph(meta_graph_or_file, clear_devices=False,
                      import_scope=None, **kwargs):
  return _import_meta_graph_with_return_elements(
      meta_graph_or_file, clear_devices, import_scope, **kwargs)[0]
def _import_meta_graph_with_return_elements(
    meta_graph_or_file, clear_devices=False, import_scope=None,
    return_elements=None, **kwargs):
    meta_graph_def = meta_graph_or_file
    imported_vars, imported_return_elements = (
      meta_graph.import_scoped_meta_graph_with_return_elements(
          meta_graph_def,
          clear_devices=clear_devices,
          import_scope=import_scope,
          return_elements=return_elements,
          **kwargs))

    saver = _create_saver_from_imported_meta_graph(
      meta_graph_def, import_scope, imported_vars) #这个函数从meta_graph_def中读取sacer_def并生成Saver（）
    return saver, imported_return_elements
```
## Saveable_Object
上文中大量涉及到了Saveable_Object,接下来就介绍一下tensorflow中的Saveable_Object
### SaveSpec类：
SaveSpec类是用来描述需要被保存的tensor切片的类
```python
class SaveSpec(object):
  def __init__(self, tensor, slice_spec, name, dtype=None):
    self._tensor = tensor 
    self.slice_spec = slice_spec #需要保存的slice
    self.name = name
 ```
 ### SaveableObject类
 该类为用于保存和重建的saveable对象的基类，该类包含四个对象：
 *op op为该类包装的对象，它提供了一个需要保存的tensor列表
 *spec: 一个SaveSpec的列表
 *name:..
 *_device
 ```python
class SaveableObject(object):
  def __init__(self, op, specs, name):
    self.op = op
    self.specs = specs
    self.name = name
    self._device = None
 ```

### ReferenceVariableSaveable(saveable_object.SaveableObject)
该类为SaveableObject的继承类，类的初始化支持var输入
```python
class ReferenceVariableSaveable(saveable_object.SaveableObject):
  def __init__(self, var, slice_spec, name):
    spec = saveable_object.SaveSpec(var, slice_spec, name, dtype=var.dtype)
    super(ReferenceVariableSaveable, self).__init__(var, [spec], name)
  def restore(self, restored_tensors, restored_shapes):
    restored_tensor = restored_tensors[0]
    if restored_shapes is not None:
      restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
    return state_ops.assign(
        self.op,
        restored_tensor,
        validate_shape=restored_shapes is None and
        self.op.get_shape().is_fully_defined()) #从restored_tensors中恢复self.op
          #为什么这个不需要赋值到设备？？


```
### ResourceVariableSaveable(saveable_object.SaveableObject)
```python
class ResourceVariableSaveable(saveable_object.SaveableObject):
    def __init__(self, var, slice_spec, name):
      self._var_device = var.device
      self._var_shape = var.shape
      if isinstance(var, ops.Tensor):  #如果var变量已经是ops.Tensor，则可以直接将handle_op和tensor记录下来
        self.handle_op = var.op.inputs[0]
        tensor = var
      elif isinstance(var, resource_variable_ops.ResourceVariable): #如果var此时是ResourceVariable则需要将var复制过来再进行下一步操作

        def _read_variable_closure(v): #这个函数主要是将不同设备上的变量拷贝到cpu0以方便记录
          def f():
            with ops.device(v.device):
              x = v.read_value()
              # To allow variables placed on non-CPU devices to be checkpointed,
              # we copy them to CPU on the same machine first.
              with ops.device("/device:CPU:0"):
                return array_ops.identity(x)
          return f

        self.handle_op = var.handle
        tensor = _read_variable_closure(var)
      else:
        raise ValueError(
            "Saveable is neither a resource variable nor a read operation."
            " Got: %s" % repr(var))
      spec = saveable_object.SaveSpec(tensor, slice_spec, name,
                                      dtype=var.dtype)
      super(ResourceVariableSaveable, self).__init__(var, [spec], name)
      
      
      
   def restore(self, restored_tensors, restored_shapes):
      restored_tensor = restored_tensors[0]
      if restored_shapes is not None:
        restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0]) #将restored_tensor进行reshape操作
      # Copy the restored tensor to the variable's device.
      with ops.device(self._var_device):   #需要将restored_tensors复制到var所在的设备上
        restored_tensor = array_ops.identity(restored_tensor)
        return resource_variable_ops.shape_safe_assign_variable_handle(
            self.handle_op, self._var_shape, restored_tensor) #调用该函数
    def saveable_op_objects_for_op(op,name): #该函数返回一个生成器，生成器中的内容是从op中生成的SaveableObject
      (特别多if 先不写了，一会儿画个流程图）
      pass
    
    def op_list_to_dict(op_list,convert_variable_to_tensor): #该函数接受一个op的列表并返回一个字典，字典的key是op的名字，value是op
      （先不写了，一会儿的。。）
    def validate_and_slice_inputs(names_to_saveables): #该函数接受一个op的字典，返回一个SaveableObject的列表
        if not isinstance(names_to_saveables, dict):
      names_to_saveables = op_list_to_dict(names_to_saveables) #这一步将list或者tuple转化成dict，dict里的元素是op

      saveables = []
      seen_ops = set()
      for name, op in sorted(names_to_saveables.items(),
                             # Avoid comparing ops, sort only by name.
                             key=lambda x: x[0]):
        for converted_saveable_object in saveable_objects_for_op(op, name):
          _add_saveable(saveables, seen_ops, converted_saveable_object)
      return saveables
      
```

  
