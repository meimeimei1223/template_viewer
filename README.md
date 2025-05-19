# Template Viewer

OpenGL を使用した 3D メッシュビューワーアプリケーション

## 機能

- 3D メッシュの可視化
- マウスコントロールによるインタラクティブなメッシュ操作
- レイキャスティングによるメッシュ交差判定
- BVH (Bounding Volume Hierarchy) による効率的な衝突検出

## 依存関係

- GLEW
- GLFW
- GLM

## ビルド方法

### Linux
```bash
mkdir build
cd build
cmake ..
make
