import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.ImmutableGraph;

import java.nio.file.Path;

public class Main {

    public static void main(String[] args) throws Exception {

        Path graphDir = GraphUtils.ensureExistingDirectory(Config.GRAPH_DIR);
        Path outDir = GraphUtils.prepareOutDirectory(Config.OUT_DIR, graphDir);

        GraphUtils.GraphBase base = GraphUtils.locateGraphBase(graphDir);
        GraphUtils.ensureWebGraphReady(graphDir, base);

        System.out.println("图目录: " + graphDir);
        System.out.println("图基名: " + base.basenamePath);
        System.out.println("输出目录: " + outDir);

        System.out.println("输出文件: "
                + Config.FILE_COLS + ", "
                + Config.FILE_OFFSETS + ", "
                + Config.FILE_NUM_NODES + ", "
                + Config.FILE_NUM_EDGES);

        System.out.println("格式: Block-Compressed CSR (delta + varint), 可 Python/C++ 解码");
        System.out.println("块大小 BLOCK_NODES = " + Config.BLOCK_NODES);
        System.out.println("字节序: little-endian（仅用于 offsets/num_*；cols 为 varint 字节流）");

        // mmap
        ImmutableGraph graph = BVGraph.loadMapped(base.basenamePath.toString());
        int numNodes = graph.numNodes();
        System.out.println("节点数(num_nodes): " + numNodes);

        // 写 BCSR
        CsrWriter writer = new CsrWriter(outDir);
        long numEdges = writer.write(graph, numNodes);

        System.out.println("边数(num_edges): " + numEdges);
        System.out.println("转换完成 ✅");
    }
}
