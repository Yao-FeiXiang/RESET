import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;
import it.unimi.dsi.webgraph.LazyIntIterator;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.*;

import static java.nio.file.StandardOpenOption.*;

public final class CsrWriter {

    private final Path outDir;

    public CsrWriter(Path outDir) {
        this.outDir = outDir;
    }

    public long write(ImmutableGraph g, int n) throws IOException {
        Path offsetsPath = outDir.resolve(Config.FILE_OFFSETS);
        Path colsPath = outDir.resolve(Config.FILE_COLS);
        Path numNodesPath = outDir.resolve(Config.FILE_NUM_NODES);
        Path numEdgesPath = outDir.resolve(Config.FILE_NUM_EDGES);

        long m = countEdges(g, n);

        writeU64(numNodesPath, n);
        writeU64(numEdgesPath, m);

        writeBlocksSequential(g, n, offsetsPath, colsPath);

        return m;
    }

    private long countEdges(ImmutableGraph g, int n) {
        long edges = 0;
        NodeIterator it = g.nodeIterator();
        while (it.hasNext()) {
            int u = it.nextInt();
            edges += it.outdegree();
            if ((u & Config.PROGRESS_NODE_MASK) == 0) {
                System.out.println("CountEdges: " + u + "/" + n + " edges=" + edges);
            }
        }
        return edges;
    }


    private void writeBlocksSequential(ImmutableGraph g, int n,
                                   Path offsetsOut, Path colsOut) throws IOException {
    final int B = Config.BLOCK_NODES;
    final int numBlocks = (n + B - 1) / B;

    try (BufferedOutputStream offsetsBos = new BufferedOutputStream(
            Files.newOutputStream(offsetsOut, CREATE, TRUNCATE_EXISTING),
            8 * 1024 * 1024
    );
         BufferedOutputStream colsBosRaw = new BufferedOutputStream(
                 Files.newOutputStream(colsOut, CREATE, TRUNCATE_EXISTING),
                 Config.OUT_BUFFER_BYTES
         )) {

        CountingOutputStream colsBos = new CountingOutputStream(colsBosRaw);

        NodeIterator it = g.nodeIterator();

        int currentBlock = 0;
        int nextBlockU = B;

        writeLongLE(offsetsBos, 0L);

        int lastU = -1;
        long writtenNodes = 0;   

        System.out.println("[BCSR-WRITE] begin");
        System.out.println("[BCSR-WRITE] expected n = " + n);
        System.out.println("[BCSR-WRITE] graph.numNodes() = " + g.numNodes());
        System.out.println("[BCSR-WRITE] B = " + B + ", numBlocks(expected) = " + numBlocks);

        while (it.hasNext()) {
            int u = it.nextInt();
            lastU = u;
            writtenNodes++;

            while (u >= nextBlockU && currentBlock < numBlocks - 1) {
                currentBlock++;
                nextBlockU += B;
                long blockStart = colsBos.bytesWritten();
                writeLongLE(offsetsBos, blockStart);
            }

            int deg = it.outdegree();
            writeUVarint(colsBos, deg);

            long prev = 0;
            LazyIntIterator succ = it.successors();
            for (int v; (v = succ.nextInt()) != -1; ) {
                long delta = (long) v - prev;
                writeZigZagVarint(colsBos, delta);
                prev = v;
            }

            if ((u & Config.PROGRESS_NODE_MASK) == 0) {
                System.out.println(
                        "[BCSR-WRITE] u=" + u +
                        " writtenNodes=" + writtenNodes +
                        " cols_bytes=" + colsBos.bytesWritten() +
                        " block=" + currentBlock + "/" + (numBlocks - 1)
                );
            }
        }

        while (currentBlock < numBlocks - 1) {
            currentBlock++;
            long blockStart = colsBos.bytesWritten();
            writeLongLE(offsetsBos, blockStart);
        }

        writeLongLE(offsetsBos, colsBos.bytesWritten());

        offsetsBos.flush();
        colsBos.flush();
        System.out.println("[BCSR-WRITE] end");
        System.out.println("[BCSR-WRITE] expected n          = " + n);
        System.out.println("[BCSR-WRITE] actual writtenNodes = " + writtenNodes);
        System.out.println("[BCSR-WRITE] lastU               = " + lastU);
        System.out.println("[BCSR-WRITE] inferred nodes      = " + (lastU + 1));
        System.out.println("[BCSR-WRITE] final block index   = " + currentBlock);
        System.out.println("[BCSR-WRITE] cols bytes total    = " + colsBos.bytesWritten());
        System.out.println("[BCSR-WRITE] ------------------------------");
    }
}

    // ===========================
    // Varint / ZigZag + LE writers
    // ===========================

    /**
     * 无符号 varint（LEB128）写入 int (>=0)
     */
    private static void writeUVarint(CountingOutputStream out, int value) throws IOException {
        long v = value & 0xFFFFFFFFL;
        while ((v & ~0x7FL) != 0) {
            out.write((int) ((v & 0x7F) | 0x80));
            v >>>= 7;
        }
        out.write((int) v);
    }

    /**
     * ZigZag 编码：把有符号 long 映射到无符号 long
     */
    private static long zigzag(long x) {
        return (x << 1) ^ (x >> 63);
    }

    /**
     * ZigZag-varint（LEB128）写入 signed long
     */
    private static void writeZigZagVarint(CountingOutputStream out, long signed) throws IOException {
        long v = zigzag(signed);
        while ((v & ~0x7FL) != 0) {
            out.write((int) ((v & 0x7F) | 0x80));
            v >>>= 7;
        }
        out.write((int) v);
    }

    /**
     * 8 字节 little-endian（用于 offsets / num_*）
     */
    private static void writeLongLE(OutputStream os, long v) throws IOException {
        for (int i = 0; i < 8; i++) {
            os.write((int) ((v >>> (8 * i)) & 0xFF));
        }
    }

    private static void writeU64(Path path, long v) throws IOException {
        try (BufferedOutputStream bos = new BufferedOutputStream(
                Files.newOutputStream(path, CREATE, TRUNCATE_EXISTING),
                64 * 1024)) {
            writeLongLE(bos, v);
        }
    }

    /**
     * 计数字节写入量的 OutputStream 包装器
     * （用于记录 block_offsets）
     */
    private static final class CountingOutputStream extends OutputStream {
        private final OutputStream delegate;
        private long bytes;

        CountingOutputStream(OutputStream delegate) {
            this.delegate = delegate;
            this.bytes = 0;
        }

        long bytesWritten() {
            return bytes;
        }

        @Override
        public void write(int b) throws IOException {
            delegate.write(b);
            bytes++;
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            delegate.write(b, off, len);
            bytes += len;
        }

        @Override
        public void flush() throws IOException {
            delegate.flush();
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }
    }
}
