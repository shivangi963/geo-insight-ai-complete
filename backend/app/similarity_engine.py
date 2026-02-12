"""
Neighborhood Similarity Engine
Finds similar neighborhoods across entire database
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NeighborhoodSimilarityEngine:
    """
    Find similar neighborhoods based on multiple factors:
    - Amenity distribution
    - Walk score
    - Density patterns
    - Price ranges (if available)
    """
    
    def __init__(self):
        self.amenity_weights = {
            'restaurant': 1.2,
            'cafe': 1.0,
            'school': 1.5,
            'hospital': 1.8,
            'park': 1.3,
            'supermarket': 1.4,
            'bank': 0.8,
            'pharmacy': 1.0,
            'gym': 0.9,
            'library': 0.7,
            'transit_station': 1.6
        }
    
    def extract_feature_vector(self, analysis: Dict) -> np.ndarray:
        """
        Extract 50-dimensional feature vector from analysis
        
        Features:
        - [0-10]: Amenity counts (normalized)
        - [11-21]: Amenity proximity scores
        - [22]: Walk score (normalized)
        - [23-32]: Amenity diversity metrics
        - [33-42]: Distance distribution
        - [43-49]: Category clusters
        """
        features = []
        
        amenities = analysis.get('amenities', {})
        walk_score = analysis.get('walk_score', 0)
        
        # 1. Amenity Counts (11 features)
        amenity_types = ['restaurant', 'cafe', 'school', 'hospital', 'park', 
                        'supermarket', 'bank', 'pharmacy', 'gym', 'library', 'transit_station']
        
        for amenity_type in amenity_types:
            items = amenities.get(amenity_type, [])
            count = len(items)
            # Normalize: 0 = none, 1 = 10+ amenities
            features.append(min(count / 10.0, 1.0))
        
        # 2. Proximity Scores (11 features)
        for amenity_type in amenity_types:
            items = amenities.get(amenity_type, [])
            if items:
                # Average distance of closest 3
                closest = sorted(items, key=lambda x: x.get('distance_km', 10))[:3]
                avg_dist = np.mean([item.get('distance_km', 5.0) for item in closest])
                # Closer = higher score
                proximity_score = max(0, 1.0 - (avg_dist / 2.0))
            else:
                proximity_score = 0.0
            features.append(proximity_score)
        
        # 3. Walk Score (1 feature)
        features.append(walk_score / 100.0)
        
        # 4. Diversity Metrics (10 features)
        total_amenities = sum(len(items) for items in amenities.values())
        features.append(min(total_amenities / 50.0, 1.0))  # Total count
        
        # How many amenity types are present
        amenity_variety = sum(1 for items in amenities.values() if len(items) > 0)
        features.append(amenity_variety / len(amenity_types))  # Diversity ratio
        
        # Shannon entropy (diversity measure)
        if total_amenities > 0:
            proportions = [len(items) / total_amenities for items in amenities.values() if len(items) > 0]
            entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
            features.append(min(entropy / 3.0, 1.0))  # Normalized entropy
        else:
            features.append(0.0)
        
        # Gini coefficient (evenness of distribution)
        if total_amenities > 0:
            counts = sorted([len(items) for items in amenities.values()])
            n = len(counts)
            gini = sum((2 * i - n - 1) * count for i, count in enumerate(counts, 1)) / (n * sum(counts))
            features.append(gini)
        else:
            features.append(0.0)
        
        # Density within 500m
        close_count = sum(1 for items in amenities.values() 
                         for item in items if item.get('distance_km', 10) < 0.5)
        features.append(min(close_count / 20.0, 1.0))
        
        # Density within 1km
        medium_count = sum(1 for items in amenities.values() 
                          for item in items if item.get('distance_km', 10) < 1.0)
        features.append(min(medium_count / 40.0, 1.0))
        
        # Weighted amenity score
        weighted_sum = sum(
            len(amenities.get(atype, [])) * self.amenity_weights.get(atype, 1.0)
            for atype in amenity_types
        )
        features.append(min(weighted_sum / 50.0, 1.0))
        
        # Essential services ratio (hospital, supermarket, pharmacy)
        essential = sum(len(amenities.get(t, [])) for t in ['hospital', 'supermarket', 'pharmacy'])
        features.append(essential / max(total_amenities, 1))
        
        # Recreation ratio (park, gym, cafe)
        recreation = sum(len(amenities.get(t, [])) for t in ['park', 'gym', 'cafe'])
        features.append(recreation / max(total_amenities, 1))
        
        # Education ratio (school, library)
        education = sum(len(amenities.get(t, [])) for t in ['school', 'library'])
        features.append(education / max(total_amenities, 1))
        
        # 5. Distance Distribution (10 features)
        all_distances = [item.get('distance_km', 5.0) 
                        for items in amenities.values() 
                        for item in items]
        
        if all_distances:
            # Percentiles
            for p in [10, 25, 50, 75, 90]:
                dist = np.percentile(all_distances, p)
                features.append(1.0 - min(dist / 3.0, 1.0))
            
            # Distance buckets
            buckets = [
                sum(1 for d in all_distances if d < 0.3),
                sum(1 for d in all_distances if 0.3 <= d < 0.7),
                sum(1 for d in all_distances if 0.7 <= d < 1.5),
                sum(1 for d in all_distances if 1.5 <= d < 2.5),
                sum(1 for d in all_distances if d >= 2.5)
            ]
            total = sum(buckets)
            features.extend([b / total if total > 0 else 0.0 for b in buckets])
        else:
            features.extend([0.0] * 10)
        
        # 6. Category Clusters (7 features)
        # Dining density
        dining = sum(len(amenities.get(t, [])) for t in ['restaurant', 'cafe'])
        features.append(min(dining / 15.0, 1.0))
        
        # Healthcare access
        healthcare = sum(len(amenities.get(t, [])) for t in ['hospital', 'pharmacy'])
        features.append(min(healthcare / 8.0, 1.0))
        
        # Transit connectivity
        transit = len(amenities.get('transit_station', []))
        features.append(min(transit / 5.0, 1.0))
        
        # Shopping convenience
        shopping = len(amenities.get('supermarket', []))
        features.append(min(shopping / 5.0, 1.0))
        
        # Green space access
        green = len(amenities.get('park', []))
        features.append(min(green / 5.0, 1.0))
        
        # Financial services
        financial = len(amenities.get('bank', []))
        features.append(min(financial / 5.0, 1.0))
        
        # Fitness facilities
        fitness = len(amenities.get('gym', []))
        features.append(min(fitness / 5.0, 1.0))
        
        return np.array(features)
    
    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate similarity between two feature vectors
        Uses cosine similarity (0-1 scale)
        """
        # Cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def find_similar_neighborhoods(
        self,
        query_analysis: Dict,
        all_analyses: List[Dict],
        limit: int = 5,
        threshold: float = 0.6,
        query_address: str = None
    ) -> List[Dict]:
        """
        Find similar neighborhoods from database
        
        Args:
            query_analysis: The reference analysis
            all_analyses: All available analyses to compare against
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of similar neighborhoods with similarity scores
        """
        query_id = query_analysis.get('id') or query_analysis.get('_id')
        
        # Extract query features
        query_vector = self.extract_feature_vector(query_analysis)
        
        # Calculate similarities
        similarities = []
        
        for analysis in all_analyses:
            # Skip self
            analysis_id = analysis.get('id') or analysis.get('_id')
            if str(analysis_id) == str(query_id):
                continue
            
            # Skip incomplete analyses
            if analysis.get('status') != 'completed':
                continue

            if query_address:
                candidate_address = analysis.get('address', '').strip().lower()
                if candidate_address == query_address:
                    continue
            
            # Extract features
            try:
                candidate_vector = self.extract_feature_vector(analysis)
                similarity = self.calculate_similarity(query_vector, candidate_vector)
                
                if similarity >= threshold:
                    similarities.append({
                        'analysis': analysis,
                        'similarity': float(similarity)
                    })
            except Exception as e:
                logger.warning(f"Error comparing analysis {analysis_id}: {e}")
                continue
        
       # ✅ Deduplicate by address - keep only best match per unique address
        seen_addresses = {}
        for item in similarities:
            addr = item['analysis'].get('address', '').strip().lower()
            if addr not in seen_addresses or item['similarity'] > seen_addresses[addr]['similarity']:
                seen_addresses[addr] = item
        similarities = list(seen_addresses.values())
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top matches
        return similarities[:limit]
    
    def create_comparison_report(
        self,
        query_analysis: Dict,
        similar_neighborhoods: List[Dict]
    ) -> Dict:
        """
        Create detailed comparison report
        """
        query_amenities = query_analysis.get('amenities', {})
        query_walk_score = query_analysis.get('walk_score', 0)
        
        report = {
            'query': {
                'address': query_analysis.get('address'),
                'walk_score': query_walk_score,
                'total_amenities': sum(len(items) for items in query_amenities.values()),
                'amenity_breakdown': {k: len(v) for k, v in query_amenities.items()}
            },
            'similar_neighborhoods': [],
            'comparison_summary': {
                'total_compared': len(similar_neighborhoods),
                'average_similarity': np.mean([n['similarity'] for n in similar_neighborhoods]) if similar_neighborhoods else 0
            }
        }
        
        for match in similar_neighborhoods:
            analysis = match['analysis']
            amenities = analysis.get('amenities', {})
            
            report['similar_neighborhoods'].append({
                'address': analysis.get('address'),
                'similarity_score': match['similarity'],
                'walk_score': analysis.get('walk_score'),
                'total_amenities': sum(len(items) for items in amenities.values()),
                'walk_score_diff': analysis.get('walk_score', 0) - query_walk_score,
                'amenity_breakdown': {k: len(v) for k, v in amenities.items()},
                'key_differences': self._identify_key_differences(query_analysis, analysis),
                'analysis_id': str(analysis.get('id') or analysis.get('_id')),
                'map_path': analysis.get('map_path'),
                'coordinates': analysis.get('coordinates')
            })
        
        return report
    
    def _identify_key_differences(self, query: Dict, candidate: Dict) -> List[str]:
        """Identify key differences between two neighborhoods"""
        differences = []
        
        query_amenities = query.get('amenities', {})
        candidate_amenities = candidate.get('amenities', {})
        
        # Walk score difference
        walk_diff = abs(query.get('walk_score', 0) - candidate.get('walk_score', 0))
        if walk_diff > 15:
            if candidate.get('walk_score', 0) > query.get('walk_score', 0):
                differences.append(f"More walkable (+{walk_diff:.0f} walk score)")
            else:
                differences.append(f"Less walkable (-{walk_diff:.0f} walk score)")
        
        # Major amenity differences
        for amenity_type in ['restaurant', 'school', 'hospital', 'park', 'transit_station']:
            query_count = len(query_amenities.get(amenity_type, []))
            candidate_count = len(candidate_amenities.get(amenity_type, []))
            diff = candidate_count - query_count
            
            if abs(diff) >= 3:
                name = amenity_type.replace('_', ' ').title()
                if diff > 0:
                    differences.append(f"More {name}s (+{diff})")
                else:
                    differences.append(f"Fewer {name}s ({diff})")
        
        return differences[:5]  # Top 5 differences


# Global instance
similarity_engine = NeighborhoodSimilarityEngine()