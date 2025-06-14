# Chord Distribution Summary

## Key Findings

Analysis of chord distribution in the HarmonySeeker AI project reveals:

### 1. Training Dataset (Isophonics Raw Annotations)

- **Major**: 68.31% (18,526.04 seconds)
- **Minor**: 24.26% (6,579.61 seconds)
- **Other**: 7.43% (2,013.53 seconds)

Top 5 most common chords:
1. A (major): 14.29%
2. G (major): 11.54%
3. D (major): 11.36%
4. E (major): 11.11%
5. C (major): 9.26%

### 2. Test Dataset

- **Major**: 69.60% (87 files)
- **Minor**: 30.40% (38 files)

Top 5 most common chords:
1. D (major): 12.80%
2. C (major): 12.00%
3. A (major): 11.20%
4. F (major): 9.60%
5. E (major): 8.80%

## Important Observations

1. **Similar Distribution Pattern**: Both training and test datasets show a similar distribution pattern, with major chords being predominant (68.31% in training, 69.60% in test).

2. **Consistent Top Chords**: The most common chords in both datasets are major chords, with A, C, and D appearing in the top 5 for both datasets.

3. **Balanced Dataset**: The chord distribution in both datasets aligns with typical music theory expectations, where major chords are more commonly used than minor chords.

4. **Measurement Differences**: 
   - Training data is measured in seconds (duration)
   - Test data is measured in number of files

## Conclusion

The chord distribution analysis confirms that both training (Isophonics) and test datasets have consistent distributions, with major chords being the predominant chord type in both cases. This consistency between training and test data provides a solid foundation for the chord recognition model.

The similar distribution pattern (approximately 68-70% major chords and 24-30% minor chords) indicates that the model should be able to generalize well from training to testing, as it won't encounter a significant distribution shift between the datasets. 