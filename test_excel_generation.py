#!/usr/bin/env python3
"""
Test script to verify Excel generation logic
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the parsing function
import importlib.util
spec = importlib.util.spec_from_file_location("vtu_automated_results", "vtu-automated-results.py")
vtu_automated_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vtu_automated_results)

def test_excel_generation():
    """Test the Excel generation logic"""
    print("Testing Excel generation logic...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    pdf_results_dir = project_root / "pdf_results"
    output_dir = project_root / "excel_results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"PDF Results Directory: {pdf_results_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Process all PDF files
    results = vtu_automated_results.process_pdf_batch(pdf_results_dir)
    
    print(f"Processed {len(results)} PDF files")
    
    # Create subject-wise DataFrame with the exact format requested
    # Collect all unique subject codes (not names)
    all_subjects = set()
    for result in results:
        for subject in result.get('subjects', []):
            all_subjects.add(subject.get('subject_code', '').strip())
    
    # Sort subject codes for consistent column ordering
    subject_columns = sorted(list(all_subjects))
    
    # Create rows for each student
    rows = []
    for result in results:
        # Clean up the student name
        student_name = result.get('student_name', '')
        if '\n' in student_name:
            student_name = student_name.split('\n')[0].strip()
        
        row = {
            'USN': result.get('usn', ''),
            'Name': student_name,  # Changed from 'Student Name' to 'Name'
        }
        
        # Add subject marks
        for subject_code in subject_columns:
            row[subject_code] = ''  # Initialize with empty string
        
        # Fill in actual subject marks
        for subject in result.get('subjects', []):
            subject_code = subject.get('subject_code', '').strip()
            if subject_code in row:
                row[subject_code] = subject.get('total', '')
        
        # Add CGPA and SGPA from result data
        row['CGPA'] = result.get('cgpa', '')
        row['SGPA'] = result.get('sgpa', '')
        
        # Determine overall result
        subject_results = [subject.get('result', '').upper() for subject in result.get('subjects', [])]
        if all(res == 'P' for res in subject_results) and subject_results:
            row['Result'] = 'P'  # Changed from 'RESULT' to 'Result'
        elif any(res == 'F' for res in subject_results):
            # Find the first failed subject
            failed_subject = None
            for subject in result.get('subjects', []):
                if subject.get('result', '').upper() == 'F':
                    failed_subject = subject.get('subject_code', '').strip()
                    break
            row['Result'] = f'F in {failed_subject}' if failed_subject else 'F'  # Changed from 'RESULT' to 'Result'
        else:
            row['Result'] = 'N/A'  # Changed from 'RESULT' to 'Result'
        
        rows.append(row)
    
    # Define column order to match exact requirements: USN, Name, subject codes, Result, CGPA, SGPA
    columns = ['USN', 'Name'] + subject_columns + ['Result', 'CGPA', 'SGPA']
    
    # Create DataFrame with the exact column structure
    subject_wise_df = pd.DataFrame(rows, columns=columns)
    
    # Display the DataFrame structure
    print("\nDataFrame structure:")
    print(f"Columns: {list(subject_wise_df.columns)}")
    print(f"Shape: {subject_wise_df.shape}")
    print("\nFirst few rows:")
    print(subject_wise_df.head())
    
    # Create Excel file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    excel_path = output_dir / f"vtu_results_test_{timestamp}.xlsx"
    subject_wise_df.to_excel(excel_path, index=False)
    print(f"\n✅ Excel file saved: {excel_path}")

if __name__ == "__main__":
    test_excel_generation()