"""
Quick test to verify the notebook has the correct joinline code
"""

import nbformat

print("="*60)
print("TESTING NOTEBOOK FOR JOINLINE FIX")
print("="*60)

# Read notebook
nb = nbformat.read('GM_OUT_Geometric_Classification.ipynb', as_version=4)

print(f"\n✅ Notebook loaded successfully")
print(f"   Total cells: {len(nb.cells)}")

# Check for joinline in cells
joinline_found = False
correct_joinline = "joinline = np.array([0, 5, 6, 7, 1, 2, 3, 4, 0])"

for i, cell in enumerate(nb.cells):
    if cell['cell_type'] == 'code' and 'joinline' in cell['source']:
        print(f"\n📍 Found 'joinline' in Cell {i}")
        
        # Check if it's the correct joinline
        if correct_joinline in cell['source']:
            print(f"   ✅ CORRECT: Uses proper joinline for gorilla skulls")
            joinline_found = True
            
            # Show the visualization code
            if 'sk_F[joinline' in cell['source']:
                print(f"   ✅ CORRECT: Plots use joinline (sk_F[joinline, 0, i])")
            else:
                print(f"   ⚠️  WARNING: joinline defined but not used in plots")
        else:
            print(f"   ⚠️  Found joinline but it's different")
        
        # Show first 300 chars of the cell
        print(f"\n   Cell preview:")
        print(f"   {cell['source'][:300]}...")

if joinline_found:
    print("\n" + "="*60)
    print("✅ SUCCESS: Notebook has correct joinline code!")
    print("="*60)
    print("\nNext steps:")
    print("1. Open Jupyter: jupyter notebook")
    print("2. Open the notebook")
    print("3. Click: Kernel → Restart & Clear Output")
    print("4. Click: Cell → Run All")
    print("5. You should see proper skull outlines!")
else:
    print("\n" + "="*60)
    print("❌ ERROR: Joinline not found or incorrect")
    print("="*60)
    print("\nRun: python create_notebook.py")
    print("Then try again.")

print("\n" + "="*60)
