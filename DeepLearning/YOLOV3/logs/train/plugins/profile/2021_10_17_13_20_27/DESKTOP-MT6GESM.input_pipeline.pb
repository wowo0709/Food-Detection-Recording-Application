  *��C�,�@�����r�@2u
>Iterator::Root::Prefetch::ParallelMapV2::BatchV2::Shuffle::Map�5Z��Q@!���ÇU@)�N"¿Q@1�]���T@:Preprocessing2g
0Iterator::Root::Prefetch::ParallelMapV2::BatchV2��p�܂T@!dF�s��X@)����a�&@1�Қ�o+@:Preprocessing2�
`Iterator::Root::Prefetch::ParallelMapV2::BatchV2::Shuffle::Map::FlatMap[0]::FlatMap[0]::TFRecord�v�w@!��{��@)�v�w@1��{��@:Advanced file read2p
9Iterator::Root::Prefetch::ParallelMapV2::BatchV2::Shuffle>?��Q@!A����U@)ճ ��q�?1��0�f�?:Preprocessing2�
SIterator::Root::Prefetch::ParallelMapV2::BatchV2::Shuffle::Map::FlatMap[0]::FlatMap�Ͻ�@!+ ��@)'��>V�?1PA��I�?:Preprocessing2~
GIterator::Root::Prefetch::ParallelMapV2::BatchV2::Shuffle::Map::FlatMap�����@!\�:�d@)�x���e�?1l�c�?:Preprocessing2O
Iterator::Root::Prefetchyxρ��?!�F'�~�?)yxρ��?1�F'�~�?:Preprocessing2^
'Iterator::Root::Prefetch::ParallelMapV2oH�'ې?! W.�5��?)oH�'ې?1 W.�5��?:Preprocessing2E
Iterator::Root\��Ϝ��?!s��&E�?)�;�ą?1ID�b�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.