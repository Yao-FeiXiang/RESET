public final class Config {
 private Config() {}

 // 数据集目录
 public static final String GRAPH_DIR = "/data4/cliu26/arabic-2005";

 // 输出目录
 public static final String OUT_DIR = "/data4/cliu26/arabic-2005/bcsr";

 // 写文件缓冲区大小
 public static final int OUT_BUFFER_BYTES = 64 * 1024 * 1024; // 64MB

 // 进度打印频率
 public static final int PROGRESS_NODE_MASK = 0xFFFFFF;

 // 分块大小（建议 2^16 = 65536；块越大索引越小但块内定位更慢）
 public static final int BLOCK_NODES = 1 << 16; // 65536

 // 输出文件名（按你的需求固定）
 public static final String FILE_COLS = "raw_csr_cols.bin";
 public static final String FILE_OFFSETS = "raw_csr_offsets.bin";
 public static final String FILE_NUM_NODES = "num_nodes.bin";
 public static final String FILE_NUM_EDGES = "num_edges.bin";
}
