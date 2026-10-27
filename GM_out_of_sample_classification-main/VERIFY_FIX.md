# ✅ How to Verify the Orientation Fix

## Quick Verification Steps

### Step 1: Regenerate Notebook (Already Done!)
```bash
cd GM_out_of_sample_classification-main
python create_notebook.py
```
✅ **Status:** COMPLETED - Notebook regenerated with fixes

### Step 2: Clear Jupyter Cache (YOU MUST DO THIS!)

**Why?** Jupyter caches old output. Even though the code is fixed, you're seeing old cached results.

**How to clear:**
1. Open `GM_OUT_Geometric_Classification.ipynb` in Jupyter
2. Click menu: `Kernel` → `Restart & Clear Output`
3. Click menu: `Cell` → `Run All`
4. Wait for all cells to execute

### Step 3: Look for Success Messages

After running all cells, you should see:

```
✅ All shapes now aligned and comparable!
📌 All individual shapes follow their group mean orientation!
```

### Step 4: Visual Verification

**What to look for in the plots:**

#### ❌ BEFORE FIX (What you were seeing):
```
Female Skulls After GPA:
  Shape 1: notch pointing ↗
  Shape 2: notch pointing ↙
  Shape 3: notch pointing ↖
  Shape 4: notch pointing ↘
  Mean:    notch pointing →
  
  ❌ Scattered orientations - shapes NOT following mean!
```

#### ✅ AFTER FIX (What you should see now):
```
Female Skulls After GPA:
  Shape 1: notch pointing →
  Shape 2: notch pointing →
  Shape 3: notch pointing →
  Shape 4: notch pointing →
  Mean:    notch pointing →
  
  ✅ Consistent orientation - all shapes follow mean!
```

### Step 5: Check Specific Plots

**Plot 1: "Female Skulls After GPA"**
- All blue shapes should have notches pointing in roughly the same direction
- Black mean shape shows the reference orientation
- Individual shapes should follow this orientation

**Plot 2: "Male Skulls After GPA"**  
- All red shapes should have notches pointing in roughly the same direction
- Black mean shape shows the reference orientation
- Individual shapes should follow this orientation

**Plot 3: "Combined Sample After GPA"**
- Blue (female) shapes: all oriented consistently with blue mean
- Red (male) shapes: all oriented consistently with red mean
- Shape differences clearly visible (not hidden by random rotations)

## Common Issues

### Issue: "I still see scattered orientations"

**Solution:** You didn't clear Jupyter cache!
1. `Kernel` → `Restart & Clear Output`
2. `Cell` → `Run All`
3. Wait for completion

### Issue: "Notebook won't load / JSON error"

**Solution:** Regenerate the notebook:
```bash
cd GM_out_of_sample_classification-main
python create_notebook.py
```

### Issue: "I see old output with old messages"

**Solution:** Jupyter is showing cached output. Clear it:
1. `Kernel` → `Restart & Clear Output`
2. `Cell` → `Run All`

## Technical Verification

If you want to verify the code changes directly:

### Check 1: rotate_skulls function uses atan2
```python
# Should see this in the notebook:
theta = np.arctan2(alpha[1], alpha[0])  # ✅ Correct

# NOT this:
theta = np.arctan(alpha[1] / alpha[0])  # ❌ Wrong
```

### Check 2: rotate_skulls returns TWO values
```python
# Should see this in the notebook:
return Rot, ms_rotated  # ✅ Correct

# NOT this:
return Rot  # ❌ Wrong
```

### Check 3: Calls unpack BOTH values
```python
# Should see this in the notebook:
Proc_F_rotated, Proc_F_mshape_rotated = rotate_skulls(...)  # ✅ Correct

# NOT this:
Proc_F_rotated = rotate_skulls(...)  # ❌ Wrong
```

## Summary

The fix is complete in the code. You just need to:
1. ✅ Regenerate notebook (DONE)
2. ⚠️ Clear Jupyter cache (YOU MUST DO THIS!)
3. ✅ Run all cells
4. ✅ Verify consistent orientations in plots

**The key:** Jupyter caches output. You MUST clear it to see the new results!
