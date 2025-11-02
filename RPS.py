import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def player(prev_play, opponent_history=[]):
    # === Global persistent initialization ===
    if not hasattr(player, "initialized"):
        player.initialized = True
        player.df_history = pd.DataFrame(columns=["context", "next"])
        player.context_window = 4  # Reduced for faster pattern detection
        player.counter_moves = {"R": "P", "P": "S", "S": "R"}
        player.possible = np.array(["R", "P", "S"])
        player.history_queue = deque(maxlen=player.context_window)
        player.my_last_move = None
        player.opponent_history = []
        player.consecutive_wins = 0
        player.consecutive_losses = 0
        
        # ML Model initialization
        player.ml_model = None
        player.label_encoder = LabelEncoder()
        player.label_encoder.fit(['R', 'P', 'S'])
        player.training_data = []
        player.training_labels = []
        player.ml_confidence_threshold = 0.60
        player.ml_update_frequency = 50  # Train more frequently
        player.round_count = 0
        
        # Pattern detection
        player.last_patterns = deque(maxlen=10)
        player.opponent_pattern_counter = Counter()

    player.round_count += 1

    # Update opponent history
    if prev_play and prev_play in ["R", "P", "S"]:
        player.opponent_history.append(prev_play)
        player.history_queue.append(prev_play)

    # For first few moves, use counter strategies
    if len(player.opponent_history) <= 2:
        if len(player.opponent_history) == 0:
            move = np.random.choice(player.possible)
            player.my_last_move = move
            return move
        
        # Counter common opening strategies
        opponent_moves = player.opponent_history
        if len(opponent_moves) == 1:
            # If opponent repeats first move, counter it
            return player.counter_moves[opponent_moves[0]]
        elif len(opponent_moves) == 2:
            # Detect patterns in first two moves
            if opponent_moves[0] == opponent_moves[1]:
                return player.counter_moves[opponent_moves[1]]
            else:
                # Counter the most recent move
                return player.counter_moves[opponent_moves[1]]

    # Build training data and update pattern database
    if len(player.opponent_history) > player.context_window:
        context = "".join(player.opponent_history[-(player.context_window + 1):-1])
        next_move = player.opponent_history[-1]
        if len(context) == player.context_window:
            player.training_data.append(list(context))
            player.training_labels.append(next_move)
            
            # Update pattern database
            player.df_history = pd.concat([
                player.df_history, 
                pd.DataFrame([{"context": context, "next": next_move}])
            ], ignore_index=True)

    # Train ML model more frequently with smaller datasets
    if (player.round_count % player.ml_update_frequency == 0 and 
        len(player.training_data) > 15):
        player.ml_model = train_ml_model(player.training_data, player.training_labels, player.label_encoder)

    # Get current context for prediction
    current_context = "".join(list(player.history_queue)) if len(player.history_queue) == player.context_window else ""

    # 1. First try ML prediction
    ml_prediction = None
    if player.ml_model is not None and current_context:
        ml_prediction = predict_with_ml(player, list(current_context))
        
        if (ml_prediction and 
            ml_prediction['confidence'] > player.ml_confidence_threshold and
            len(player.opponent_history) > 10):  # Only trust ML after some history
            
            predicted_opponent_move = ml_prediction['move']
            counter_move = player.counter_moves[predicted_opponent_move]
            
            # Adaptive learning based on ML performance
            if hasattr(player, 'last_ml_prediction'):
                last_predicted = player.last_ml_prediction.get('move')
                actual_move = player.opponent_history[-1] if player.opponent_history else None
                if last_predicted and actual_move and last_predicted == actual_move:
                    player.consecutive_wins += 1
                else:
                    player.consecutive_wins = max(0, player.consecutive_wins - 1)
            
            player.last_ml_prediction = ml_prediction
            
            # If ML is performing well, use it
            if player.consecutive_wins >= 2:
                player.my_last_move = counter_move
                return counter_move

    # 2. Pattern-based prediction with improved logic
    pattern_prediction = enhanced_pattern_prediction(player, current_context)
    if pattern_prediction:
        counter_move = player.counter_moves[pattern_prediction]
        player.my_last_move = counter_move
        return counter_move

    # 3. Fallback: Counter the most frequent opponent move
    if len(player.opponent_history) > 5:
        freq_prediction = frequency_based_prediction(player)
        counter_move = player.counter_moves[freq_prediction]
        player.my_last_move = counter_move
        return counter_move

    # 4. Ultimate fallback with anti-pattern logic
    final_move = np.random.choice(player.possible)
    player.my_last_move = final_move
    return final_move

def enhanced_pattern_prediction(player, context):
    """Improved pattern prediction with multiple strategies"""
    if not context or len(player.df_history) == 0:
        return None
    
    # Strategy 1: Exact context match
    exact_matches = player.df_history[player.df_history["context"] == context]
    if len(exact_matches) >= 2:
        next_moves = exact_matches["next"].value_counts()
        if len(next_moves) > 0:
            return next_moves.index[0]
    
    # Strategy 2: Partial context matching (progressive)
    for length in range(player.context_window - 1, 1, -1):
        partial_context = context[-length:]
        partial_matches = player.df_history[
            player.df_history["context"].str.endswith(partial_context)
        ]
        if len(partial_matches) >= 3:  # Require more matches for partial contexts
            next_moves = partial_matches["next"].value_counts()
            if len(next_moves) > 0:
                return next_moves.index[0]
    
    # Strategy 3: Opponent tendency detection
    if len(player.opponent_history) >= 10:
        recent_moves = player.opponent_history[-8:]
        move_counts = Counter(recent_moves)
        most_common_move = move_counts.most_common(1)[0][0]
        
        # If opponent shows strong tendency, counter it
        if move_counts[most_common_move] >= len(recent_moves) * 0.6:
            return most_common_move
    
    return None

def frequency_based_prediction(player):
    """Predict based on opponent's move frequency"""
    if not player.opponent_history:
        return np.random.choice(player.possible)
    
    recent_moves = player.opponent_history[-10:]  # Look at recent moves
    move_counts = Counter(recent_moves)
    
    # Consider the opponent's counter to our last move
    if player.my_last_move and len(player.opponent_history) >= 2:
        my_second_last = player.my_last_move
        opponent_response = player.opponent_history[-1]
        
        # If opponent consistently counters us, predict they'll do it again
        similar_situations = []
        for i in range(1, min(10, len(player.opponent_history))):
            if (i < len(player.opponent_history) and 
                player.my_last_move == my_second_last):
                similar_situations.append(player.opponent_history[i])
        
        if similar_situations:
            response_counts = Counter(similar_situations)
            if len(response_counts) > 0:
                most_common_response = response_counts.most_common(1)[0][0]
                return most_common_response
    
    # Default to most frequent recent move
    return move_counts.most_common(1)[0][0]

def train_ml_model(training_data, training_labels, label_encoder):
    """Train an improved ML model"""
    try:
        if len(training_data) < 10:
            return None
            
        X = []
        for context in training_data:
            features = []
            for move in context:
                if move == 'R':
                    features.extend([1, 0, 0])
                elif move == 'P':
                    features.extend([0, 1, 0])
                elif move == 'S':
                    features.extend([0, 0, 1])
            X.append(features)
        
        y = label_encoder.transform(training_labels)
        
        # Improved Random Forest with better parameters
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt'
        )
        model.fit(X, y)
        
        return model
    except Exception:
        return None

def predict_with_ml(player, current_context):
    """Use ML model for prediction"""
    try:
        if player.ml_model is None:
            return None
            
        features = []
        for move in current_context:
            if move == 'R':
                features.extend([1, 0, 0])
            elif move == 'P':
                features.extend([0, 1, 0])
            elif move == 'S':
                features.extend([0, 0, 1])
        
        probabilities = player.ml_model.predict_proba([features])[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        predicted_move = player.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return {'move': predicted_move, 'confidence': confidence}
    except Exception:
        return None