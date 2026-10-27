# 📐 Joinline Explained

## What is a Joinline?

A **joinline** is an array that defines the **order in which landmarks should be connected** to visualize the shape outline.

## Why Do We Need It?

Landmarks are just points in space. Without a joinline, we don't know which points connect to which, resulting in messy visualizations like you saw.

## Example: Gorilla Skull (8 Landmarks)

### Landmark Positions:
```
Landmark 0: Top-left
Landmark 1: Top-middle-left  
Landmark 2: Top-middle-right
Landmark 3: Top-right
Landmark 4: Bottom-right
Landmark 5: Bottom-left
Landmark 6: Left-middle-top
Landmark 7: Left-middle-bottom
```

### Joinline for Gorilla Skull:
```python
joinline = np.array([0, 5, 6, 7, 1, 2, 3, 4, 0])
```

This means:
1. Start at landmark 0
2. Connect to landmark 5
3. Connect to landmark 6
4. Connect to landmark 7
5. Connect to landmark 1
6. Connect to landmark 2
7. Connect to landmark 3
8. Connect to landmark 4
9. Connect back to landmark 0 (closes the loop)

### Visual Representation:
```
    0 -------- 1 -------- 2 -------- 3
    |                                 |
    |                                 |
    5                                 4
    |                                 |
    |                                 |
    6 -------- 7 -------- (back) ----+
```

## How to Use in Plotting

### Without Joinline (Wrong):
```python
# This connects landmarks in order 0→1→2→3→4→5→6→7
plt.plot(shape[:, 0], shape[:, 1], 'b-')
# Result: Messy, incorrect connections
```

### With Joinline (Correct):
```python
# This connects landmarks in the order specified by joinline
plt.plot(shape[joinline, 0], shape[joinline, 1], 'b-')
# Result: Proper skull outline
```

## For Your Arm Data

When you get arm contour data, you'll need to define your own joinline based on how the landmarks are arranged.

### Example: Arm with 8 Landmarks

If your arm landmarks are arranged like this:
```
    0 (top-left)
    |
    7 (left-middle)
    |
    6 (bottom-left)
    |
    5 (bottom-middle)
    |
    4 (bottom-right)
    |
    3 (right-middle)
    |
    2 (top-right)
    |
    1 (top-middle)
```

Your joinline might be:
```python
joinline = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0])
```

## How to Determine Your Joinline

1. **Look at your landmark numbering scheme**
2. **Trace the outline** of your shape
3. **Write down the landmark numbers** in the order you trace
4. **Add the first landmark again** at the end to close the loop

## Common Patterns

### Closed Loop (like skull or arm outline):
```python
joinline = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0])
# Starts at 0, goes around, returns to 0
```

### Open Curve (like a line):
```python
joinline = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# Starts at 0, ends at 7, doesn't close
```

### Complex Shape (with branches):
```python
# Main outline
joinline_main = np.array([0, 1, 2, 3, 4, 5, 0])
# Branch
joinline_branch = np.array([2, 6, 7])
```

## In the Notebook

The notebook now uses:
```python
joinline = np.array([0, 5, 6, 7, 1, 2, 3, 4, 0])
```

This is specifically for gorilla skull data from the R `shapes` package.

**When you switch to arm data, update this to match your landmark arrangement!**

## Quick Test

To verify your joinline is correct:
```python
# Plot one shape with joinline
plt.figure(figsize=(8, 8))
plt.plot(shape[joinline, 0], shape[joinline, 1], 'bo-', linewidth=2, markersize=8)
plt.title('Check if outline looks correct')
plt.axis('equal')
plt.grid(True)
plt.show()
```

If the outline looks correct, your joinline is good! ✅

## Summary

| Concept | Description | Example |
|---------|-------------|---------|
| **Landmarks** | Points in space | `(x, y)` coordinates |
| **Joinline** | Connection order | `[0, 5, 6, 7, 1, 2, 3, 4, 0]` |
| **Purpose** | Visualize shape outline | Proper skull/arm contour |
| **Customization** | Match your data | Update for arm landmarks |

The joinline is just a visualization aid - it doesn't affect the Procrustes analysis or classification! 🎯
