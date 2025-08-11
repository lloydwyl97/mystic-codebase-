from ai_auto_learner import AIAutoLearner
from simulation_logger import SimulationLogger


def get_ai_rating():
    logger = SimulationLogger()
    learner = AIAutoLearner()
    summary = logger.get_summary()

    score = 50  # base score

    score += summary["avg_profit"] * 10
    score += learner.state["confidence_threshold"] * 25
    score -= learner.state["adjustment_count"]

    score = max(0, min(100, round(score)))

    if score >= 80:
        rank = "A+ (Excellent)"
    elif score >= 60:
        rank = "B (Good)"
    elif score >= 40:
        rank = "C (Needs Improvement)"
    else:
        rank = "D (Unstable)"

    return {
        "ai_score": score,
        "rating": rank,
        "adjustments": learner.state["adjustment_count"],
        "avg_profit": summary["avg_profit"],
        "confidence_threshold": learner.state["confidence_threshold"],
    }


def get_ai_health_report():
    rating = get_ai_rating()
    logger = SimulationLogger()
    summary = logger.get_summary()

    health_indicators = {
        "performance": "good" if summary["avg_profit"] > 0 else "poor",
        "stability": "stable" if rating["adjustments"] < 5 else "unstable",
        "confidence": ("high" if rating["confidence_threshold"] > 0.8 else "low"),
        "activity": "active" if summary["total_trades"] > 10 else "inactive",
    }

    return {
        "rating": rating,
        "health_indicators": health_indicators,
        "recommendations": generate_recommendations(rating, summary),
    }


def generate_recommendations(rating, summary):
    recommendations = []

    if rating["avg_profit"] < 0:
        recommendations.append("Consider lowering confidence threshold")

    if rating["adjustments"] > 10:
        recommendations.append("AI may be over-adjusting - consider reset")

    if summary["total_trades"] < 5:
        recommendations.append("Need more trading data for accurate assessment")

    if not recommendations:
        recommendations.append("AI performing well - continue current strategy")

    return recommendations
