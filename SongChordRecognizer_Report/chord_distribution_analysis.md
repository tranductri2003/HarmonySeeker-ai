# Detailed Chord Distribution Analysis

## 1. Training Dataset (Isophonics Raw Annotations)

Analysis of the `.lab` annotation files in the Isophonics dataset reveals:

### Overall Distribution

| Chord Type | Percentage | Duration |
|------------|------------|----------|
| Major      | 68.31%     | 18,526.04 seconds |
| Minor      | 24.26%     | 6,579.61 seconds |
| Other      | 7.43%      | 2,013.53 seconds |

### Detailed Distribution (Top 10)

| Chord  | Percentage | Duration (seconds) |
|--------|------------|-------------------|
| A      | 14.29%     | 3,874.13 |
| G      | 11.54%     | 3,129.50 |
| D      | 11.36%     | 3,079.62 |
| E      | 11.11%     | 3,012.79 |
| C      | 9.26%      | 2,511.11 |
| A:min  | 4.81%      | 1,304.48 |
| B      | 4.02%      | 1,090.70 |
| E:min  | 3.71%      | 1,007.42 |
| F      | 3.67%      | 994.69 |
| D:min  | 3.09%      | 839.09 |

## 2. Test Dataset

Analysis based on the number of audio files in each chord directory:

### Overall Distribution

| Chord Type | Percentage | Number of Files |
|------------|------------|-----------------|
| Major      | 69.60%     | 87 files |
| Minor      | 30.40%     | 38 files |

### Detailed Distribution (Top 10)

| Chord  | Percentage | Number of Files |
|--------|------------|-----------------|
| D      | 12.80%     | 16 |
| C      | 12.00%     | 15 |
| A      | 11.20%     | 14 |
| F      | 9.60%      | 12 |
| E      | 8.80%      | 11 |
| G      | 7.20%      | 9 |
| A:min  | 7.20%      | 9 |
| E:min  | 5.60%      | 7 |
| C:min  | 4.00%      | 5 |
| B:min  | 4.00%      | 5 |

## 3. Comparative Analysis

### 3.1 Similarities

1. **Major Chord Dominance**: Both datasets show a clear predominance of major chords:
   - Training: 68.31% major
   - Test: 69.60% major

2. **Top Chords Consistency**: The most common chords in both datasets are major chords, with A, C, D, and E appearing in the top 5 for both datasets.

3. **Similar Minor Chord Representation**: Minor chords represent approximately one-fourth to one-third of all chords in both datasets:
   - Training: 24.26% minor
   - Test: 30.40% minor

### 3.2 Differences

1. **Measurement Units**:
   - Training data is measured in seconds (duration)
   - Test data is measured in number of files

2. **Specific Chord Rankings**:
   - A is the most common chord in training (14.29%)
   - D is the most common chord in test (12.80%)

3. **Minor Chord Distribution**:
   - Test dataset has slightly higher proportion of minor chords (30.40% vs 24.26%)

## 4. Music Theory Context

The distribution observed in both datasets aligns with music theory expectations:

1. **Major Chord Prevalence**: Major chords are generally more common in Western music, which is reflected in both datasets.

2. **Popular Keys**: The most common chords (A, G, D, C, E) correspond to commonly used keys in popular music.

3. **Balanced Representation**: The datasets provide a balanced representation of both major and minor chords, with appropriate proportions that reflect real-world music composition practices.

## 5. Implications for Model Training

### 5.1 Advantages

1. **Consistent Distribution**: The similarity between training and test distributions suggests that the model should generalize well.

2. **Natural Balance**: The natural distribution of chords in music is preserved, allowing the model to learn realistic patterns.

3. **Adequate Minor Representation**: While minor chords are less common, they still have sufficient representation for the model to learn their characteristics.

### 5.2 Considerations

1. **Duration vs Count**: The training data measures chord duration while the test data counts files, which might affect how the model perceives chord importance.

2. **Rare Chords**: Some chords have very low representation in both datasets, which might affect the model's ability to recognize them accurately.

## 6. Conclusion

The chord distribution analysis confirms that both training and test datasets have consistent and musically appropriate distributions. The predominance of major chords (approximately 68-70%) in both datasets reflects typical music composition patterns and provides a solid foundation for training a chord recognition model.

The similarity between training and test distributions suggests that the model should be able to generalize well from training to testing, as it won't encounter a significant distribution shift between the datasets. 