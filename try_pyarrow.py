import pyarrow.parquet as pq
import pyarrow as pa
import pathlib
_path = pathlib.Path("abc").resolve()
_path /= "xyz"
_path /= ".lmn"  # uncomment this has a 'dot' at start of the folder
_path /= "file_name"
pq.write_to_dataset(
    table=pa.table({"a": [1], "b": [2]}),
    root_path=_path.as_posix()
)
t = pq.read_table(_path)
print(t.to_pandas())