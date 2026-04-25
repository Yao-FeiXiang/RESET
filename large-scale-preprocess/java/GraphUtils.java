import it.unimi.dsi.webgraph.BVGraph;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.*;
import java.util.Comparator;

public final class GraphUtils {

    private GraphUtils() {}

    public static final class GraphBase {
        public final String baseName;
        public final Path basenamePath;

        GraphBase(String baseName, Path basenamePath) {
            this.baseName = baseName;
            this.basenamePath = basenamePath;
        }
    }

    public static Path ensureExistingDirectory(String path) {
        Path p = Paths.get(path).toAbsolutePath().normalize();
        if (!Files.isDirectory(p)) {
            throw new IllegalStateException("不是有效目录: " + p);
        }
        return p;
    }

    public static Path prepareOutDirectory(String outDir, Path graphDir) throws IOException {
        if (outDir == null || outDir.isBlank()) return graphDir;
        Path p = Paths.get(outDir).toAbsolutePath().normalize();
        Files.createDirectories(p);
        return p;
    }

    // 找到 .graph 文件并解析 basename
    public static GraphBase locateGraphBase(Path dir) throws IOException {
        Path graphFile;
        try (var s = Files.list(dir)) {
            graphFile = s.filter(p -> p.getFileName().toString().endsWith(".graph"))
                    .min(Comparator.comparing(p -> p.getFileName().toString()))
                    .orElseThrow(() -> new FileNotFoundException("未找到 .graph 文件: " + dir));
        }

        String name = graphFile.getFileName().toString();
        String baseName = name.substring(0, name.length() - ".graph".length());
        return new GraphBase(baseName, dir.resolve(baseName));
    }

    public static void ensureWebGraphReady(Path dir, GraphBase base) throws IOException {
        // 确保 properties / offsets 就绪
        Path prop = dir.resolve(base.baseName + ".properties");
        if (!Files.exists(prop)) {
            throw new FileNotFoundException("缺少 .properties 文件: " + prop);
        }

        Path offsets = dir.resolve(base.baseName + ".offsets");
        if (Files.exists(offsets)) return;

        System.out.println("未检测到 offsets，开始构建...");
        try {
            BVGraph.main(new String[]{"-o", "-O", "-L", base.basenamePath.toString()});
        } catch (Exception e) {
            System.out.println("降级为仅 -o 构建 offsets");
            try {
                BVGraph.main(new String[]{"-o", base.basenamePath.toString()});
            } catch (Exception e2) {
                throw new RuntimeException("offsets 构建失败", e2);
            }
        }

        if (!Files.exists(offsets)) {
            throw new IOException("offsets 构建后仍不存在: " + offsets);
        }
    }
}
