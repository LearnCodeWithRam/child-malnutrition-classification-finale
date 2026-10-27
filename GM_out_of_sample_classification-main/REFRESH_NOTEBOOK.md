# 🔄 How to Refresh the Notebook and See New Plots

## The Problem

You're seeing old plots because Jupyter has cached the previous output. The notebook code is correct, but you need to clear the cache and re-run.

## ✅ Solution: 3 Easy Steps

### Step 1: Close the Notebook

In Jupyter, close the `GM_OUT_Geometric_Classification.ipynb` tab.

### Step 2: Restart Jupyter (Optional but Recommended)

In your terminal/command prompt:
1. Press `Ctrl+C` to stop Jupyter
2. Restart: `jupyter notebook`

### Step 3: Reopen and Run

1. Open `GM_OUT_Geometric_Classification.ipynb`
2. Click **Kernel → Restart & Clear Output**
3. Click **Cell → Run All**

## 🎯 What You Should See Now

### Before (Old - Wrong):
- Landmarks connected in random order (0→1→2→3...)
- Zigzag lines that don't follow skull outline

### After (New - Correct):
- Landmarks connected in proper order using joinline
- Smooth skull outlines
- Landmarks shown as dots on the outline
- Mean shapes clearly visible

## 🔍 Verify the Fix

After running, check Cell 7 output. You should see:
```
📊 Raw data shows variation in position, size, and orientation
📌 Joinline order: [0 5 6 7 1 2 3 4 0] (connects landmarks to form outline)
```

If you see this message, the joinline is working! ✅

## 🚨 Still Seeing Old Plots?

### Option A: Clear All Output Manually

1. In Jupyter: **Cell → All Output → Clear**
2. Then: **Cell → Run All**

### Option B: Delete and Regenerate Notebook

```bash
cd GM_out_of_sample_classification-main

# Delete old notebook
rm GM_OUT_Geometric_Classification.ipynb

# Regenerate with correct code
python create_notebook.py

# Open fresh notebook
jupyter notebook GM_OUT_Geometric_Classification.ipynb
```

### Option C: Use JupyterLab (Better Caching)

```bash
pip install jupyterlab
jupyter lab
```

JupyterLab handles caching better than classic Jupyter Notebook.

## 📊 Expected Visualization

With the correct joinline, you should see:

**Female Skulls:**
```
    Landmark 0 ─── Landmark 5
         │              │
    Landmark 7 ─── Landmark 6
         │              │
    Landmark 1 ─── Landmark 2
         │              │
    Landmark 3 ─── Landmark 4
```

Connected in order: 0→5→6→7→1→2→3→4→0 (closes loop)

## 🎯 Quick Test

Run this in a new cell to verify joinline is defined:

```python
print(f"Joinline: {joinline}")
print(f"Connects landmarks in this order: {' → '.join(map(str, joinline))}")
```

Expected output:
```
Joinline: [0 5 6 7 1 2 3 4 0]
Connects landmarks in this order: 0 → 5 → 6 → 7 → 1 → 2 → 3 → 4 → 0
```

## ✅ Checklist

- [ ] Closed old notebook tab
- [ ] Restarted Jupyter (optional)
- [ ] Opened notebook fresh
- [ ] Clicked "Kernel → Restart & Clear Output"
- [ ] Clicked "Cell → Run All"
- [ ] Verified joinline message appears
- [ ] Plots now show proper skull outlines

If all checked, you should see the correct visualizations! 🎉

## 💡 Pro Tip

To avoid this in the future:
- Always use **"Kernel → Restart & Run All"** when reopening notebooks
- This ensures you're seeing fresh output, not cached results

---

**The notebook code is correct - you just need to clear the cache and re-run!** ✅
