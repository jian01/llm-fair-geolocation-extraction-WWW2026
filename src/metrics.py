from .dataset import Dataset

def false_positive_rate(predicted_tags: Dataset, ground_truth: Dataset):
    false_positives = 0
    total = 0
    for k in predicted_tags.keys():
        if not k in ground_truth:
            raise ValueError("A predicted document is not in the ground truth")
        total += predicted_tags[k].unique_tags
        for tag in predicted_tags[k].tags:
            if tag not in ground_truth[k].tags:
                false_positives += 1
    return false_positives/total

def false_negative_rate(predicted_tags: Dataset, ground_truth: Dataset):
    false_negatives = 0
    total = 0
    for k in ground_truth.keys():
        if not k in predicted_tags:
            raise ValueError("A ground truth document is not in the predicted tags")
        total += ground_truth[k].unique_tags
        prev_end = 0
        sorted_tags = sorted(ground_truth[k].tags, key=lambda x: -len(x.text))
        sorted_tags = sorted(sorted_tags, key=lambda x: x.start)
        for t in sorted_tags:
            if t in predicted_tags[k].tags:
                prev_end = t.end
            if t.start > prev_end and t not in predicted_tags[k].tags:
                false_negatives += 1
    return false_negatives/total
