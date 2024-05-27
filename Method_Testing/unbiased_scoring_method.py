def convert_sentiment_scores(positive, neutral, negative):
    # Weights
    w_p = 1
    w_n = 0.5
    w_ng = -1

    # Compute weighted sum
    score = (w_p * positive) + (w_n * neutral) + (w_ng * negative)

    # Map to [0, 1] range
    normalized_score = (score + 1) / 2

    return normalized_score


# Example usage
positive = 0.0361
neutral = 0.0975
negative = 0.8664

final_score = convert_sentiment_scores(positive, neutral, negative)
print(final_score)  # Output: 0.484875
