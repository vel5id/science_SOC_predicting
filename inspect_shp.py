"""Inspect shapefile schemas across all years/farms to understand column naming."""
from pathlib import Path
import struct

def read_dbf_fields(dbf_path: Path) -> list[str]:
    """Read field names from a .dbf file (lightweight, no dependencies)."""
    fields = []
    with open(dbf_path, 'rb') as f:
        # DBF header
        version = struct.unpack('B', f.read(1))[0]
        f.read(3)  # date
        num_records = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<H', f.read(2))[0]
        record_size = struct.unpack('<H', f.read(2))[0]
        f.read(20)  # reserved
        
        # Field descriptors (32 bytes each), header ends with 0x0D
        num_fields = (header_size - 33) // 32
        for _ in range(num_fields):
            field_data = f.read(32)
            if field_data[0] == 0x0D:
                break
            name = field_data[:11].split(b'\x00')[0].decode('ascii', errors='replace')
            ftype = chr(field_data[11])
            fsize = field_data[16]
            fields.append(f"{name}({ftype}/{fsize})")
    return fields, num_records


def main():
    root = Path(r"c:/Claude/science_article/raw_data/shp")
    
    all_schemas = {}
    
    for shp_file in sorted(root.rglob("*.shp")):
        dbf_file = shp_file.with_suffix('.dbf')
        if not dbf_file.exists():
            continue
        
        rel = shp_file.relative_to(root)
        parts = rel.parts
        year = parts[0]
        farm = parts[1]
        field = parts[2] if len(parts) > 2 else "?"
        
        try:
            fields, n_records = read_dbf_fields(dbf_file)
            key = f"{year}/{farm}/{field}"
            all_schemas[key] = {"fields": fields, "records": n_records}
        except Exception as e:
            print(f"ERROR reading {rel}: {e}")
    
    # Group by unique schema
    schema_groups: dict[str, list[str]] = {}
    for key, info in all_schemas.items():
        schema_key = ", ".join(info["fields"])
        if schema_key not in schema_groups:
            schema_groups[schema_key] = []
        schema_groups[schema_key].append(f"{key} ({info['records']} rows)")
    
    print(f"Total shapefiles: {len(all_schemas)}")
    print(f"Unique schemas: {len(schema_groups)}")
    print("=" * 80)
    
    for i, (schema, files) in enumerate(schema_groups.items(), 1):
        print(f"\n--- Schema {i} ---")
        print(f"Fields: {schema}")
        print(f"Files ({len(files)}):")
        for f in files[:5]:
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more")


if __name__ == "__main__":
    main()
