import pefile
import pandas as pd
import argparse
import os

def extract_pe_header_features(pe):
    features = {}
    features["e_magic"] = pe.DOS_HEADER.e_magic
    features["e_cblp"] = pe.DOS_HEADER.e_cblp
    features["e_cp"] = pe.DOS_HEADER.e_cp
    features["e_crlc"] = pe.DOS_HEADER.e_crlc
    features["e_cparhdr"] = pe.DOS_HEADER.e_cparhdr
    features["e_minalloc"] = pe.DOS_HEADER.e_minalloc
    features["e_maxalloc"] = pe.DOS_HEADER.e_maxalloc
    features["e_ss"] = pe.DOS_HEADER.e_ss
    features["e_sp"] = pe.DOS_HEADER.e_sp
    features["e_csum"] = pe.DOS_HEADER.e_csum
    features["e_ip"] = pe.DOS_HEADER.e_ip
    features["e_cs"] = pe.DOS_HEADER.e_cs
    features["e_lfarlc"] = pe.DOS_HEADER.e_lfarlc
    features["e_ovno"] = pe.DOS_HEADER.e_ovno
    features["NumberOfSections"] = pe.FILE_HEADER.NumberOfSections
    features["TimeDateStamp"] = pe.FILE_HEADER.TimeDateStamp
    features["PointerToSymbolTable"] = pe.FILE_HEADER.PointerToSymbolTable
    features["NumberOfSymbols"] = pe.FILE_HEADER.NumberOfSymbols
    features["SizeOfOptionalHeader"] = pe.FILE_HEADER.SizeOfOptionalHeader
    features["Characteristics"] = pe.FILE_HEADER.Characteristics
    features["MajorLinkerVersion"] = pe.OPTIONAL_HEADER.MajorLinkerVersion
    features["MinorLinkerVersion"] = pe.OPTIONAL_HEADER.MinorLinkerVersion
    features["SizeOfCode"] = pe.OPTIONAL_HEADER.SizeOfCode
    features["SizeOfInitializedData"] = pe.OPTIONAL_HEADER.SizeOfInitializedData
    features["SizeOfUninitializedData"] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
    features["AddressOfEntryPoint"] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    features["BaseOfCode"] = pe.OPTIONAL_HEADER.BaseOfCode
    features["ImageBase"] = pe.OPTIONAL_HEADER.ImageBase
    features["SectionAlignment"] = pe.OPTIONAL_HEADER.SectionAlignment
    features["FileAlignment"] = pe.OPTIONAL_HEADER.FileAlignment
    features["SizeOfImage"] = pe.OPTIONAL_HEADER.SizeOfImage
    features["SizeOfHeaders"] = pe.OPTIONAL_HEADER.SizeOfHeaders
    features["Subsystem"] = pe.OPTIONAL_HEADER.Subsystem
    features["DllCharacteristics"] = pe.OPTIONAL_HEADER.DllCharacteristics
    return features

def extract_section_features(pe):
    section_data = {}
    for section in pe.sections:
        name = section.Name.decode(errors='ignore').strip('\x00')
        prefix = name.lower()
        section_data[f"{prefix}_VirtualSize"] = section.Misc_VirtualSize
        section_data[f"{prefix}_VirtualAddress"] = section.VirtualAddress
        section_data[f"{prefix}_SizeOfRawData"] = section.SizeOfRawData
        section_data[f"{prefix}_PointerToRawData"] = section.PointerToRawData
        section_data[f"{prefix}_Characteristics"] = section.Characteristics
    return section_data

def extract_dll_features(pe):
    dll_features = {}
    try:
        dlls = [entry.dll.decode().lower() for entry in pe.DIRECTORY_ENTRY_IMPORT]
        for dll in dlls:
            dll_features[dll] = 1
    except AttributeError:
        pass
    return dll_features

def main(file_path, output_csv):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        pe = pefile.PE(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse PE file: {e}")
        return

    print("[INFO] Extracting features...")
    header_features = extract_pe_header_features(pe)
    section_features = extract_section_features(pe)
    dll_features = extract_dll_features(pe)

    combined = {}
    combined.update(header_features)
    combined.update(section_features)
    combined.update(dll_features)

    df = pd.DataFrame([combined])
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Features extracted and saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a PE file")
    parser.add_argument("exe_path", help="Path to the .exe file")
    parser.add_argument("output_csv", help="Output .csv file to save extracted features")
    args = parser.parse_args()

    main(args.exe_path, args.output_csv)
