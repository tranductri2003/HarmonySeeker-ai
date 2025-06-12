class KeyRecognizer:

    key_chords = {
        # C major, D minor, E minor, F major, G major, A minor, B minor
        "C": [1, 6, 10, 11, 15, 20, 24],
        # Cis/Des major, Dis/Es minor, F minor, Fis/Ges major, Gis/As major, Ais/Bes minor, C minor
        "C#/Db": [3, 8, 12, 13, 17, 22, 2],
        # D major, E minor, Fis/Ges minor, G major, A major, B minor, Cis/Des minor
        "D": [5, 10, 14, 15, 19, 24, 4],
        # Dis/Es major, F minor, G minor, Gis/As major, Ais/Bes major, C minor, D minor
        "D#/Eb": [7, 12, 16, 17, 21, 2, 6],
        # E major, Fis/Ges minor, Gis/As minor, A major, B major, Cis/Des minor, Dis/Es minor
        "E": [9, 14, 18, 19, 23, 4, 8],
        # F major, G minor, A minor, Ais/Bes major, C major, D minor, E minor
        "F": [11, 16, 20, 21, 1, 6, 10],
        # Fis/Ges major, Gis/Des minor, Ais/Bes minor, B major, Cis/Des major, Dis/Es minor, F minor
        "F#/Gb": [13, 18, 22, 23, 3, 8, 12],
        # G major, A minor, B minor, C major, D major, E minor, Fis/Ges minor
        "G": [15, 20, 24, 1, 5, 10, 14],
        # Gis/As major, Ais/Bes minor, C minor, Cis/Des major, Dis/Es major, F minor, G minor
        "G#/Ab": [17, 22, 2, 3, 7, 12, 16],
        # A major, B minor, Cis/Des minor, D major, E major, Fis/Ges minor, Gis/As minor
        "A": [19, 24, 4, 5, 9, 14, 18],
        # Ais/Bes major, C minor, D minor, Dis/Es major, F major, G minor, A minor
        "A#/Bb": [21, 2, 6, 7, 11, 16, 20],
        # B major, Cis/Des minor, Dis/Es minor, E major, Fis/Ges major, Gis/As minor, Ais/Bes minor
        "B": [23, 4, 8, 9, 13, 18, 22],
    }

    # Explicit mappings between major and minor keys
    major_to_minor_map = {
        "C": "A:min",
        "C#": "A#:min",
        "D": "B:min",
        "D#": "C:min",
        "E": "C#:min",
        "F": "D:min",
        "F#": "D#:min",
        "G": "E:min",
        "G#": "F:min",
        "A": "F#:min",
        "A#": "G:min",
        "B": "G#:min",
    }

    minor_to_major_map = {
        "A:min": "C",
        "A#:min": "C#",
        "B:min": "D",
        "C:min": "D#",
        "C#:min": "E",
        "D:min": "F",
        "D#:min": "F#",
        "E:min": "G",
        "F:min": "G#",
        "F#:min": "A",
        "G:min": "A#",
        "G#:min": "B",
    }

    @staticmethod
    def estimate_key(chord_counts, target_scale=None, use_relative_mode=False):
        """
        Estimate the musical key of a song based on predicted chord distribution.

        By default, this function returns the most likely major key, determined by
        counting how many chords match typical chords of each key (based on `key_chords`).

        If `use_relative_mode=True` and `target_scale` is provided ('major' or 'minor'),
        the result will be mapped to the corresponding relative key. For example:
        - If the estimated key is 'C' and `target_scale='minor'`, the result will be 'A:min'.
        - If the estimated key is 'A:min' and `target_scale='major'`, the result will be 'C'.

        Parameters
        ----------
        chord_counts : dict[int, int]
            A dictionary where the key is the chord index and the value is the count
            of how often the chord appeared in the song.
        target_scale : str, optional
            If set to 'major' or 'minor', the final result will be mapped accordingly
            when `use_relative_mode=True`.
        use_relative_mode : bool, optional
            Whether to apply relative key mapping based on the target scale.

        Returns
        -------
        key : str
            The estimated musical key of the song (e.g., 'C', 'A:min', 'G#:min', etc.).
        """
        scores = {}
        for key in KeyRecognizer.key_chords:
            score = 0
            for chord_ind in KeyRecognizer.key_chords[key]:
                if chord_ind in chord_counts:
                    score += chord_counts[chord_ind]
            scores[key] = score

        # Sort descending
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_key = sorted_keys[0][0].split("/")[0]
        if best_key == "N" and len(sorted_keys) > 1:
            best_key = sorted_keys[1][0].split("/")[0]

        if not use_relative_mode or not target_scale:
            return best_key

        target_scale = target_scale.lower()
        is_best_minor = best_key.endswith(":min")
        is_target_minor = target_scale == "minor"

        # Case 1: Same scale type
        if is_best_minor and is_target_minor:
            return best_key
        if not is_best_minor and not is_target_minor:
            return best_key

        # Case 2: Convert from major to minor
        if not is_best_minor and is_target_minor:
            if best_key in KeyRecognizer.major_to_minor_map:
                return KeyRecognizer.major_to_minor_map[best_key]
            else:
                print(f"[WARN] No mapping from major to minor for: {best_key}")
                return best_key + ":min"

        # Case 3: Convert from minor to major
        if is_best_minor and not is_target_minor:
            if best_key in KeyRecognizer.minor_to_major_map:
                return KeyRecognizer.minor_to_major_map[best_key]
            else:
                print(f"[WARN] No mapping from minor to major for: {best_key}")
                return best_key.replace(":min", "")

        return best_key
