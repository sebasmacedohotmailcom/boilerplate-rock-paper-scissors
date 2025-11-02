# Rock Paper Scissors AI Agent

A machine learning project for the freeCodeCamp "Rock Paper Scissors" challenge that creates an AI agent capable of achieving competitive win rates against various computer opponents.

## 🧩 Project Description
This is my solution to the freeCodeCamp **Machine Learning with Python - Rock Paper Scissors Project**.  
The goal is to create an AI player that can play Rock-Paper-Scissors against different computer opponents and achieve a win rate of at least **60%**.

---

## ⚙️ My Approach

### 🧠 Multi-Strategy Architecture

#### Machine Learning Component
- **Random Forest Classifier** trained on opponent move patterns  
- One-hot encoding of move sequences for feature engineering  
- Dynamic retraining every 50 rounds  
- Confidence thresholding (**60% minimum confidence**)  

#### 🔍 Pattern Recognition System
- Exact context matching with **4-move sequences**  
- Progressive partial matching when exact matches fail  
- Real-time pattern database that grows during gameplay  
- Move tendency detection for opponent behavior analysis  

#### 📊 Statistical Analysis
- Frequency-based prediction using recent move counts  
- Opponent response pattern detection  
- Consecutive move analysis for streak detection  

---

## 🧠 Core Parameters
```python
context_window = 4            # Pattern sequence length
ml_confidence_threshold = 0.60  # Minimum ML confidence
ml_update_frequency = 50      # Rounds between retraining
```

---

## 📈 Performance

### Overall Results
✅ Meets freeCodeCamp requirement: **>60% win rate** against most opponents  
🎯 Best performance: **Up to 61% win rate** in optimal conditions  
🔄 Consistent results across multiple sessions  

### Current Challenge: *Abby Opponent*
**Performance Issue:** Agent struggles to maintain target 60% win rate (usually 59%-61%).

#### Analysis
- Abby uses adaptive counter-strategies against pattern prediction  
- ML confidence thresholds might be too conservative  
- Pattern recognition may fail for complex decision trees  

#### Potential Solutions
- Dynamic confidence threshold adjustment  
- Enhanced pattern window and matching logic  
- Opponent-specific strategy selection  
- Meta-learning for opponent type detection  

---

## 🛠️ Installation & Setup

### Requirements
```bash
pip install tensorflow numpy pandas scikit-learn
```

### Project Structure
```
rock-paper-scissors/
├── RPS.py                 # Main game file (provided by freeCodeCamp)
├── RPS_Tournament.py      # Tournament runner (provided by freeCodeCamp)
└── my_agent.py            # This AI agent implementation
```

---

## 🚀 Usage

### Basic Implementation
```python
def player(prev_play, opponent_history=[]):
    # Multi-strategy AI implementation
    # Returns: "R", "P", or "S"
```

### Testing
```python
# Test against a single opponent
from my_agent import player
play(player, opponent, games=1000)
```

### Tournament Testing
```bash
python RPS_Tournament.py
```

---

## 💡 Code Implementation Highlights

- **State Management:** Persistent game state across function calls  
- **ML Training:** Incremental Random Forest model with engineered features  
- **Pattern Database:** Real-time collection and matching of move sequences  
- **Adaptive Learning:** Strategy adjustment based on recent performance  

### Prediction Hierarchy
1. **Primary:** ML prediction (confidence > 60%)  
2. **Secondary:** Pattern recognition  
3. **Tertiary:** Frequency-based analysis  
4. **Fallback:** Strategic random choice  

---

## 🔄 Development Process

| Stage | Description |
|-------|--------------|
| **Initial Approach** | Basic counter-strategies and frequency analysis |
| **ML Integration** | Added Random Forest classifier |
| **Multi-layer System** | Combined ML + pattern matching + stats |
| **Optimization** | Tuned parameters and thresholds |

---

## 🔮 Future Improvements

### Short-term Goals
- Dynamic confidence threshold adjustment for Abby  
- Enhanced ML feature engineering  
- Opponent classification system  
- Pattern window optimization  

### Technical Enhancements
- Neural network implementation  
- Multi-step prediction planning  
- Ensemble modeling  
- Real-time hyperparameter optimization  

---

## ✅ freeCodeCamp Requirements Met

- ✅ Function signature: `player(prev_play, opponent_history=[])`  
- ✅ Achieves >60% win rate against most opponents  
- ✅ Uses machine learning (Random Forest)  
- ✅ Maintains persistent game state  

---

## 🪪 License
This project is created for **educational purposes** as part of the **freeCodeCamp Machine Learning with Python** certification.

**Author:** Created for freeCodeCamp Machine Learning with Python Certification
