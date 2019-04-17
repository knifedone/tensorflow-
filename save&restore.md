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
    
      
