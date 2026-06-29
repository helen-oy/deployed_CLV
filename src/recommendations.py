"""
Recommendation engine module for CLV prediction system.
Generates personalized recommendations based on customer segments and CLV.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generates personalized recommendations for customers based on their segments and CLV.
    """

    def __init__(self):
        # Define recommendation strategies for each segment combination
        self.recommendation_matrix = {
            ('Champions', 'Low'): [
                "VIP loyalty rewards program",
                "Exclusive early access to new products",
                "Personalized thank you communications",
                "Premium customer support"
            ],
            ('Champions', 'Moderate'): [
                "Upsell premium products",
                "Cross-sell complementary items",
                "Referral program incentives",
                "Exclusive member events"
            ],
            ('Champions', 'High'): [
                "Immediate retention campaign",
                "Personal concierge service",
                "Custom product bundles",
                "Loyalty program reactivation"
            ],
            ('Loyal', 'Low'): [
                "Loyalty program rewards",
                "Personalized product recommendations",
                "Birthday/anniversary specials",
                "Exclusive member discounts"
            ],
            ('Loyal', 'Moderate'): [
                "Upsell opportunities",
                "Cross-sell recommendations",
                "Engagement campaigns",
                "Loyalty tier upgrades"
            ],
            ('Loyal', 'High'): [
                "Retention incentives",
                "Personal outreach",
                "Custom retention packages",
                "Loyalty program reactivation"
            ],
            ('Active', 'Low'): [
                "Re-engagement campaigns",
                "Personalized offers",
                "Product discovery recommendations",
                "Loyalty program enrollment"
            ],
            ('Active', 'Moderate'): [
                "Upsell campaigns",
                "Cross-sell opportunities",
                "Engagement incentives",
                "Personalized communications"
            ],
            ('Active', 'High'): [
                "Win-back campaigns",
                "Discounted retention offers",
                "Personal outreach",
                "Re-engagement incentives"
            ],
            ('Occasional', 'Low'): [
                "Re-engagement emails",
                "Special promotional offers",
                "Product recommendations",
                "Simplified purchase process"
            ],
            ('Occasional', 'Moderate'): [
                "Promotional campaigns",
                "Discount incentives",
                "Cross-sell opportunities",
                "Engagement building"
            ],
            ('Occasional', 'High'): [
                "Reactivation campaigns",
                "Deep discount offers",
                "Personal outreach",
                "Simplified re-engagement"
            ],
            ('Low', 'Low'): [
                "Basic re-engagement",
                "Generic promotional offers",
                "Product awareness campaigns",
                "Low-cost incentives"
            ],
            ('Low', 'Moderate'): [
                "Targeted promotional offers",
                "Discount campaigns",
                "Product recommendations",
                "Re-engagement incentives"
            ],
            ('Low', 'High'): [
                "Aggressive reactivation",
                "Deep discount promotions",
                "Personal outreach",
                "High-touch re-engagement"
            ]
        }

    def _segment_clv(self, clv_values: pd.Series) -> pd.Series:
        """Create robust CLV bands even when many customers share the same value."""
        segment_labels = ['Low', 'Occasional', 'Active', 'Loyal', 'Champions']
        valid_clv = clv_values.replace([np.inf, -np.inf], np.nan).fillna(0)
        quantiles = valid_clv.quantile([0.2, 0.4, 0.6, 0.8]).tolist()

        bin_edges = [-np.inf]
        for edge in quantiles:
            if edge > bin_edges[-1]:
                bin_edges.append(edge)
        bin_edges.append(np.inf)

        usable_labels = segment_labels[:len(bin_edges) - 1]
        return pd.cut(
            valid_clv,
            bins=bin_edges,
            labels=usable_labels,
            include_lowest=True
        )

    def segment_customers_by_clv_and_churn(self, clv_predictions: pd.DataFrame,
                                         churn_probabilities: pd.Series) -> pd.DataFrame:
        """
        Segment customers based on CLV and churn risk.

        Args:
            clv_predictions: DataFrame with CLV predictions
            churn_probabilities: Series with churn probabilities

        Returns:
            DataFrame with segment assignments
        """
        logger.info("Segmenting customers by CLV and churn risk")

        # Create combined DataFrame
        segments = clv_predictions[['CustomerID', 'CLV']].copy()
        segments['churn_probability'] = churn_probabilities

        # CLV segmentation (quantiles)
        segments['CLV_Segment'] = self._segment_clv(segments['CLV'])

        # Churn risk segmentation
        segments['Churn_Risk'] = pd.cut(
            segments['churn_probability'],
            bins=[-np.inf, 0.2, 0.5, np.inf],
            labels=['Low', 'Moderate', 'High'],
            include_lowest=True
        )

        segments['CLV_Segment'] = segments['CLV_Segment'].astype(object).fillna('Low')
        segments['Churn_Risk'] = segments['Churn_Risk'].astype(object).fillna('High')

        # Combined segment
        segments['Combined_Segment'] = segments.apply(
            lambda row: f"{row['CLV_Segment']}_{row['Churn_Risk']}", axis=1
        )

        logger.info(f"Created segments for {len(segments)} customers")
        return segments

    def generate_recommendations(self, customer_segments: pd.DataFrame) -> pd.DataFrame:
        """
        Generate personalized recommendations for each customer.

        Args:
            customer_segments: DataFrame with customer segments

        Returns:
            DataFrame with recommendations
        """
        logger.info("Generating personalized recommendations")

        recommendations = customer_segments.copy()

        # Generate recommendations based on CLV and Churn segments
        recommendations['Recommendations'] = recommendations.apply(
            self._get_recommendations_for_customer, axis=1
        )

        # Add priority score (higher CLV + higher churn risk = higher priority)
        recommendations['Priority_Score'] = (
            recommendations['CLV'].rank(pct=True) +
            recommendations['churn_probability'].rank(pct=True)
        ) / 2

        # Sort by priority
        recommendations = recommendations.sort_values('Priority_Score', ascending=False)

        logger.info("Recommendations generated successfully")
        return recommendations

    def _get_recommendations_for_customer(self, customer_row: pd.Series) -> List[str]:
        """
        Get recommendations for a specific customer based on their segments.

        Args:
            customer_row: Row containing customer segment information

        Returns:
            List of recommendation strings
        """
        clv_segment = customer_row['CLV_Segment']
        churn_risk = customer_row['Churn_Risk']

        # Get recommendations from matrix
        key = (clv_segment, churn_risk)
        recommendations = self.recommendation_matrix.get(key, [])

        # If no specific recommendations, use default
        if not recommendations:
            recommendations = [
                "General re-engagement campaign",
                "Product awareness communications",
                "Basic promotional offers"
            ]

        return recommendations

    def get_segment_summary(self, customer_segments: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each segment combination.

        Args:
            customer_segments: DataFrame with customer segments

        Returns:
            DataFrame with segment summaries
        """
        summary = customer_segments.groupby(['CLV_Segment', 'Churn_Risk']).agg({
            'CustomerID': 'count',
            'CLV': ['mean', 'median', 'sum'],
            'churn_probability': 'mean'
        }).round(2)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={
            'CustomerID_count': 'customer_count',
            'CLV_mean': 'avg_clv',
            'CLV_median': 'median_clv',
            'CLV_sum': 'total_clv',
            'churn_probability_mean': 'avg_churn_risk'
        })

        return summary

    def get_top_priority_customers(self, recommendations: pd.DataFrame,
                                 top_n: int = 100) -> pd.DataFrame:
        """
        Get top priority customers for immediate action.

        Args:
            recommendations: DataFrame with recommendations
            top_n: Number of top customers to return

        Returns:
            DataFrame with top priority customers
        """
        return recommendations.head(top_n)

    def export_recommendations(self, recommendations: pd.DataFrame,
                             file_path: str) -> None:
        """
        Export recommendations to CSV file.

        Args:
            recommendations: DataFrame with recommendations
            file_path: Path to save the CSV file
        """
        recommendations.to_csv(file_path, index=False)
        logger.info(f"Recommendations exported to {file_path}")

    def create_action_plan(self, recommendations: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create an action plan organized by segment and priority.

        Args:
            recommendations: DataFrame with recommendations

        Returns:
            Dictionary with action plan
        """
        action_plan = {}

        # Group by combined segment
        for segment, group in recommendations.groupby('Combined_Segment'):
            # Sort by priority within segment
            top_customers = group.head(10)['CustomerID'].tolist()
            action_plan[segment] = top_customers

        return action_plan
