# -*- coding: utf-8 -*-
"""
Master Builder Script for SAM Photo Diagnosis Methodology Pipeline Notebook
Executes Steps 1 to 5, pre-computes outputs & figures, and saves the final .ipynb file.
"""
import sys
import os

# Add scratch dir to path
scratch_dir = os.path.dirname(os.path.abspath(__file__))
if scratch_dir not in sys.path:
    sys.path.insert(0, scratch_dir)

from nb_common import save_notebook
from step1_cells import build_step1
from step2_cells import build_step2
from step3_cells import build_step3
from step4_cells import build_step4
from step5_cells import build_step5

def main():
    print("="*80)
    print("STARTING COMPLETE NOTEBOOK GENERATION PIPELINE")
    print("="*80)
    
    build_step1()
    build_step2()
    build_step3()
    build_step4()
    build_step5()
    
    save_notebook()
    print("="*80)
    print("MASTER NOTEBOOK GENERATION COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
