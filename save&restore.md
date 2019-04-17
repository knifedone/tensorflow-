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
### BuldSaverBuilder()
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



  
