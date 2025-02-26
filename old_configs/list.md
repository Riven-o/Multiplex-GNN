# config files
* Mew_fuse_graph_embed.yaml: fusing pooled graph embeddings from neighboring graphs to the
central missing graphs. To be specific, if we miss graph type A, we use type B embeddings to
find neighbors, and aggregate type A embedding to the missing graphs.