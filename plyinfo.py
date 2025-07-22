from plyfile import PlyData

# 读取 ply
ply = PlyData.read("/data/dataset_45_new/13/point_clouds/0/0.ply")
                    

# 打印原生 header
print(ply.header)

# 遍历每种 element（vertex, face, …）
for elem in ply.elements:
    props = [p.name for p in elem.properties]
    print(f"Element '{elem.name}': count={elem.count}, properties={props}")
    
    if elem.name == 'vertex':
        vertex_data = elem.data
        print(f"\n--- 打印前 3 个点的属性 ---")
        for i in range(min(3, elem.count)):
            print(f"Point {i}:")
            for prop in props:
                print(f"  {prop}: {vertex_data[i][prop]}")
            print("-" * 40)
    
