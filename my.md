
# Technical Report on Periodic Segment Detection in Motion Data

## Abstract
This paper presents a method for segmenting motion data based on periodicity detection, aiming to automatically identify strongly periodic intervals within the data and further subdivide these intervals for more precise segmentation. The method employs a combination of Fast Fourier Transform (FFT) and Autocorrelation Function (ACF) techniques to detect significant periodic patterns and refine the segmentation. Experiments conducted on various types of motion data demonstrate the effectiveness of the method, and parameter tuning suggestions are provided to enhance its performance in practical applications.

## data analysis

xxxxxxxxxxx

分段的意义和方法：

### 基于数据两种方法：

#### 方法 for 周期的

xxxx



#### 方法 for 阈值


#### 

xxxx 


## Methods
We begin by reading the motion data from record files and converting the timestamps into standard datetime formats. The preprocessing steps include handling missing values and outliers to ensure data integrity and accuracy. Additionally, normalization is applied to the data to facilitate subsequent periodicity analysis.
The FFT-based periodicity detection is implemented by segmenting the data using a sliding window with a specified window_size and step_size, performing FFT on each window to compute the spectral amplitude, calculating the ratio of the peak amplitude to the total amplitude (peak_ratio) for each window as an indicator of periodicity strength, and identifying windows with significant periodicity based on a threshold (peak_ratio_threshold).
The identified strong periodic windows may overlap or be adjacent, necessitating merging. The process involves sorting the strong periodic windows by their start positions, iterating through the sorted list and merging overlapping or adjacent windows into larger intervals, and filtering out intervals that do not meet periodicity requirements.
Within the identified strong periodic intervals, further subdivision is performed using ACF and peak/trough detection. The steps involve computing the ACF values for each strong periodic interval to determine the primary period, using peak/trough detection algorithms to identify extreme points in the speed series, which typically correspond to the start and end of movement cycles, and segmenting the strong periodic intervals into smaller subintervals based on the positions of these extreme points, with each subinterval representing a complete movement cycle.

## fur


运动数据周期性分段检测技术报告


摘要

本文介绍了一种基于周期性检测的运动数据分段方法，旨在自动识别数据中的强周期性区间，并进一步细分这些区间以实现更精确的分段。该方法结合了快速傅里叶变换（FFT）和自相关函数（ACF）技术，用于检测显著的周期性模式并优化分段。在各种类型的运动数据上进行的实验表明了该方法的有效性，并提供了参数调整建议以提升其在实际应用中的性能。


方法


我们首先从记录文件中读取运动数据，并将时间戳转换为标准的日期时间格式。数据预处理步骤包括处理缺失值和异常值，以确保数据的完整性和准确性。此外，对数据进行归一化处理，以便于后续的周期性分析。
基于FFT的周期性检测通过使用指定的window_size和step_size对数据进行滑动窗口分段，对每个窗口执行FFT以计算频谱振幅，计算每个窗口的峰值振幅与总振幅的比值（peak_ratio）作为周期性强度的指标，并根据阈值（peak_ratio_threshold）识别具有显著周期性的窗口来实现。
识别出的强周期性窗口可能存在重叠或相邻的情况，需要进行合并。该过程包括按起始位置对强周期性窗口进行排序，遍历排序后的列表并将重叠或相邻的窗口合并为更大的区间，以及筛选出不符合周期性要求的区间。
在识别出的强周期性区间内，进一步使用ACF和峰谷检测进行细分。步骤包括计算每个强周期性区间的ACF值以确定主要周期，使用峰谷检测算法识别速度序列中的极值点（这些点通常对应于运动周期的开始和结束），并根据这些极值点的位置将强周期性区间细分为更小的子区间，每个子区间代表一个完整的运动周期。