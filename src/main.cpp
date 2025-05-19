#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ShaderProgram.h"

int gWindowWidth = 1024, gWindowHeight = 768;
GLFWwindow* gWindow = NULL;
bool gWireframe = false;

glm::vec3 objPos = glm::vec3(0.0f, 0.0f, 0.0f);
glm::mat4 model(1.0), view(1.0), projection(1.0);
glm::vec3 hit_position;
int hit_index;
bool isDragging,isHit;

bool cutting = false,cutMode = false, meshcut  = false,cutModeInit = false;
bool deformModeInit = false,deformMode = true, restore = false, mousePressed = false;

void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode);
void glfw_OnFramebufferSize(GLFWwindow* window, int width, int height);
void glfw_onMouseMoveOrbit(GLFWwindow* window, double posX, double posY);
void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
bool initOpenGL();
void showFPS(GLFWwindow* window);


struct Cam{
    glm::vec3 cameraPos = glm::vec3(0.0f, 3.0f, 4.0f),cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f),up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraDirection = cameraPos - cameraTarget;
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
    glm::vec3 cameraUp = glm::normalize(glm::cross(cameraDirection, cameraRight));
    float MOVE_SPEED = 3.0,MOUSE_SENSITIVITY = 0.2f,LIGHT_MOUSE_SENSITIVITY = 0.01f,gRadius = 10.0f, gYaw = 0.0f, gPitch = 0.0f,gFOV = 45.0f;
    const double ZOOM_SENSITIVITY = -1;

    void UpdateOrbit()
    {
        float Yaw = glm::radians(gYaw);
        float Pitch = glm::radians(gPitch);
        Pitch = glm::clamp(Pitch, -glm::pi<float>() / 2.0f + 0.1f,
                           glm::pi<float>() / 2.0f - 0.1f);
        gRadius = glm::clamp(gRadius, 2.0f, 80.0f);
        cameraPos.x = cameraTarget.x + gRadius * cosf(Pitch) * sinf(Yaw);
        cameraPos.y = cameraTarget.y + gRadius * sinf(Pitch);
        cameraPos.z = cameraTarget.z + gRadius * cosf(Pitch) * cosf(Yaw);
        view = glm::lookAt(cameraPos, cameraTarget, up);
        model = glm::translate(model, cameraTarget);
        projection = glm::perspective(glm::radians(gFOV),
                                      float(gWindowWidth) / float(gWindowHeight), 0.1f, 100.0f);
        cameraDirection = glm::normalize(cameraPos - cameraTarget);
        cameraRight = glm::normalize(glm::cross(up, cameraDirection));
        cameraUp = glm::normalize(glm::cross(cameraDirection, cameraRight));
    }
}OrbitCam;

// Raycast class
class RayCast {
public:
    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;
    };

    struct RayHit {
        bool hit;
        float distance;
        glm::vec3 position;
    };

    // スクリーン座標からレイを生成
    static Ray screenToRay(float screenX, float screenY,
                           const glm::mat4& view,
                           const glm::mat4& projection,
                           const glm::vec4& viewport) {
        // スクリーン座標を正規化デバイス座標(NDC)に変換
        float ndcX = (2.0f * screenX) / viewport.z - 1.0f;
        float ndcY = 1.0f - (2.0f * screenY) / viewport.w;  // Y座標は反転する

        // クリップ空間での近面と遠面の点を作成
        glm::vec4 nearPoint = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
        glm::vec4 farPoint = glm::vec4(ndcX, ndcY, 1.0f, 1.0f);

        // ビュープロジェクション行列の逆行列を計算
        glm::mat4 invVP = glm::inverse(projection * view);

        // ワールド空間に変換
        glm::vec4 worldNear = invVP * nearPoint;
        glm::vec4 worldFar = invVP * farPoint;

        // w成分で除算
        worldNear /= worldNear.w;
        worldFar /= worldFar.w;

        Ray ray;
        ray.origin = glm::vec3(worldNear);
        ray.direction = glm::normalize(glm::vec3(worldFar - worldNear));

        return ray;
    }

    // 一般的なメッシュ（vectors）との交差判定
    static RayHit intersectMesh(const Ray& ray, std::vector<GLfloat> vertices, std::vector<GLuint> indices) {
        RayHit result = { false, std::numeric_limits<float>::max(), glm::vec3(0)};

        const auto& positions = vertices;
        const auto& surfaceTriIds = indices;

        std::cout << "Ray origin: " << ray.origin.x << ", " << ray.origin.y << ", " << ray.origin.z << std::endl;
        std::cout << "Ray direction: " << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << std::endl;

        float t, u, v;
        for (size_t i = 0; i < surfaceTriIds.size(); i += 3) {
            int idx1 = surfaceTriIds[i];
            int idx2 = surfaceTriIds[i + 1];
            int idx3 = surfaceTriIds[i + 2];

            // モデル空間での頂点位置をそのまま使用
            glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
            glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
            glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);

            if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
                if (t < result.distance) {
                    result.hit = true;
                    result.distance = t;
                    result.position = ray.origin + ray.direction * t;
                    hit_index = i/3;  // グローバル変数または関数のメンバー変数として定義する必要あり
                    std::cout << "Hit at triangle " << i/3 << std::endl;
                    std::cout << "Hit position: " << result.position.x << ", "
                              << result.position.y << ", " << result.position.z << std::endl;
                }
            }
        }
        return result;
    }

private:
    // レイと三角形の交差判定(Möller–Trumboreアルゴリズム)
    static bool rayTriangleIntersect(
        const glm::vec3& rayOrigin,
        const glm::vec3& rayDir,
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        float& t,
        float& u,
        float& v) {

        const float EPSILON = 0.0000001f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(rayDir, edge2);
        float a = glm::dot(edge1, h);

        if (a > -EPSILON && a < EPSILON) return false;

        float f = 1.0f / a;
        glm::vec3 s = rayOrigin - v0;
        u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        glm::vec3 q = glm::cross(s, edge1);
        v = f * glm::dot(rayDir, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        t = f * glm::dot(edge2, q);
        return t > EPSILON;
    }
};

void FindHit(float screenX, float screenY, const std::vector<GLfloat> vertices, const std::vector<GLuint> indices) {
    // ワールド空間でのレイを生成
    view = glm::lookAt(OrbitCam.cameraPos, OrbitCam.cameraTarget, OrbitCam.up);
    projection = glm::perspective(glm::radians(OrbitCam.gFOV),
                                  float(gWindowWidth) / float(gWindowHeight), 0.1f, 100.0f);
    RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                 glm::vec4(0, 0, gWindowWidth, gWindowHeight));

    // レイはワールド空間のままで交差判定を行う
    RayCast::RayHit hit = RayCast::intersectMesh(worldRay, vertices, indices);

    std::cout << "Hit test with ray: origin=("
              << worldRay.origin.x << "," << worldRay.origin.y << "," << worldRay.origin.z
              << "), dir=(" << worldRay.direction.x << "," << worldRay.direction.y
              << "," << worldRay.direction.z << ")" << std::endl;

    std::cout << "Hit: " << (hit.hit ? "Yes" : "No") << std::endl;

    if (hit.hit) {
        std::cout << "Hit distance: " << hit.distance << std::endl;
        std::cout << "Hit position: " << hit.position.x << ", "
                  << hit.position.y << ", " << hit.position.z << std::endl;

        // ヒット位置はすでにワールド座標
        glm::vec3 hitPos = hit.position;

        // 最も近い頂点を効率的に見つける（距離の二乗を使用）
        float minD2 = std::numeric_limits<float>::max();
        int closestVertexIndex = -1;

        // 全頂点の中から最も近い頂点を1回のループで見つける
        for (size_t i = 0; i < vertices.size() / 3; i++) {
            glm::vec3 vertexPos(
                vertices[i * 3],
                vertices[i * 3 + 1],
                vertices[i * 3 + 2]
                );

            // 距離の二乗を計算（平方根計算を避けるため）
            glm::vec3 diff = vertexPos - hitPos;
            float d2 = glm::dot(diff, diff);

            if (d2 < minD2) {
                minD2 = d2;
                closestVertexIndex = i;
            }
        }

        // 最も近い頂点が見つかった場合
        if (closestVertexIndex >= 0) {
            hit_position = glm::vec3(
                vertices[closestVertexIndex * 3],
                vertices[closestVertexIndex * 3 + 1],
                vertices[closestVertexIndex * 3 + 2]
                );

            hit_index = closestVertexIndex;

            std::cout << "Closest vertex: " << closestVertexIndex
                      << " at position: " << hit_position.x << ", "
                      << hit_position.y << ", " << hit_position.z
                      << " (distance squared: " << minD2 << ")" << std::endl;
        } else {
            hit_position = hitPos;
            hit_index = -1;
        }

        isDragging = true;
    } else {
        hit_index = -1;
        isDragging = false;
    }

    std::cout << "hit_index: " << hit_index << std::endl;
}

struct mCutMesh {
    GLuint VAO, VBO, EBO,NBO;
    std::vector<GLfloat> mVertices;
    std::vector<GLfloat> mNormals;
    std::vector<GLuint> mIndices;
    int numFaces;
    glm::vec3 mColor;

    // メッシュを描画
    void draw(ShaderProgram& shader, float alpha) {
        // カラーのuniformを設定
        shader.use();
        // メインプログラムの修正
        shader.setUniform("model", model); // モデル行列をシェーダーに送信
        shader.setUniform("lightPos", OrbitCam.cameraPos);
        shader.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
        shader.setUniform("view", view);
        shader.setUniform("projection", projection);
        shader.setUniform("vertColor", glm::vec4(mColor,alpha));

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, mIndices.size(), GL_UNSIGNED_INT, 0);
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "OpenGL error during drawing: " << err << std::endl;
        }
        glBindVertexArray(0);
    }

    // クリーンアップ
    void cleanup() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &NBO);
        glDeleteBuffers(1, &EBO);
    }

    // BVHノードの構造体
    struct BVHNode {
        glm::vec3 min;         // 境界ボックスの最小点
        glm::vec3 max;         // 境界ボックスの最大点
        std::vector<uint32_t> triangleIndices;  // 含まれる三角形のインデックス
        BVHNode* left;         // 左子ノード
        BVHNode* right;        // 右子ノード

        BVHNode() : min(FLT_MAX), max(-FLT_MAX), left(nullptr), right(nullptr) {}
        ~BVHNode() {
            delete left;
            delete right;
        }
    };

    // BVH木を構築する関数
    BVHNode* buildBVH(const mCutMesh& mesh, const std::vector<uint32_t>& triangleIndices, int depth = 0) {
        const int MAX_DEPTH = 10;  // 最大深度
        const int MIN_TRIANGLES = 10;  // リーフノードの最小三角形数

        BVHNode* node = new BVHNode();

        // ノードの境界ボックスを計算
        for (uint32_t triIdx : triangleIndices) {
            for (int i = 0; i < 3; i++) {
                uint32_t vertIdx = mesh.mIndices[triIdx * 3 + i];
                glm::vec3 pos(mesh.mVertices[vertIdx * 3],
                              mesh.mVertices[vertIdx * 3 + 1],
                              mesh.mVertices[vertIdx * 3 + 2]);

                node->min = glm::min(node->min, pos);
                node->max = glm::max(node->max, pos);
            }
        }

        // 終了条件: 深すぎる、または三角形が少なすぎる
        if (depth >= MAX_DEPTH || triangleIndices.size() <= MIN_TRIANGLES) {
            node->triangleIndices = triangleIndices;
            return node;
        }

        // 最長軸に沿って分割
        int axis = 0;
        float axisLength = node->max.x - node->min.x;
        if (node->max.y - node->min.y > axisLength) {
            axis = 1;
            axisLength = node->max.y - node->min.y;
        }
        if (node->max.z - node->min.z > axisLength) {
            axis = 2;
        }

        // 軸の中点
        float splitPos = (node->min[axis] + node->max[axis]) * 0.5f;

        // 三角形を2つのグループに分割
        std::vector<uint32_t> leftTriangles, rightTriangles;

        for (uint32_t triIdx : triangleIndices) {
            // 三角形の中心を計算
            glm::vec3 center(0.0f);
            for (int i = 0; i < 3; i++) {
                uint32_t vertIdx = mesh.mIndices[triIdx * 3 + i];
                center += glm::vec3(mesh.mVertices[vertIdx * 3],
                                   mesh.mVertices[vertIdx * 3 + 1],
                                   mesh.mVertices[vertIdx * 3 + 2]);
            }
            center /= 3.0f;

            // 分割位置で振り分け
            if (center[axis] < splitPos) {
                leftTriangles.push_back(triIdx);
            } else {
                rightTriangles.push_back(triIdx);
            }
        }

        // 分割に失敗した場合（すべての三角形が片方に寄った場合）
        if (leftTriangles.empty() || rightTriangles.empty()) {
            node->triangleIndices = triangleIndices;
            return node;
        }

        // 子ノードを再帰的に構築
        node->left = buildBVH(mesh, leftTriangles, depth + 1);
        node->right = buildBVH(mesh, rightTriangles, depth + 1);

        return node;
    }

    // レイと境界ボックスの交差判定
    bool rayBoxIntersect(const glm::vec3& origin, const glm::vec3& dir,
                        const glm::vec3& boxMin, const glm::vec3& boxMax) {
        float tmin = -FLT_MAX;
        float tmax = FLT_MAX;

        for (int i = 0; i < 3; i++) {
            if (std::abs(dir[i]) < 1e-6) {
                // レイが軸に平行
                if (origin[i] < boxMin[i] || origin[i] > boxMax[i])
                    return false;
            } else {
                float invD = 1.0f / dir[i];
                float t1 = (boxMin[i] - origin[i]) * invD;
                float t2 = (boxMax[i] - origin[i]) * invD;

                if (t1 > t2) std::swap(t1, t2);

                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);

                if (tmin > tmax) return false;
            }
        }

        return true;
    }

    // BVHを使ったレイと三角形メッシュの交差判定
    int rayMeshIntersect(const glm::vec3& origin, const glm::vec3& dir,
                         const mCutMesh& mesh, BVHNode* node) {
        // 境界ボックスとの交差判定
        if (!rayBoxIntersect(origin, dir, node->min, node->max)) {
            return 0;
        }

        // リーフノードなら三角形と交差判定
        if (node->left == nullptr && node->right == nullptr) {
            int intersectionCount = 0;

            for (uint32_t triIdx : node->triangleIndices) {
                // 三角形の頂点インデックス
                uint32_t i0 = mesh.mIndices[triIdx * 3];
                uint32_t i1 = mesh.mIndices[triIdx * 3 + 1];
                uint32_t i2 = mesh.mIndices[triIdx * 3 + 2];

                // 三角形の頂点
                glm::vec3 v0(mesh.mVertices[i0 * 3], mesh.mVertices[i0 * 3 + 1], mesh.mVertices[i0 * 3 + 2]);
                glm::vec3 v1(mesh.mVertices[i1 * 3], mesh.mVertices[i1 * 3 + 1], mesh.mVertices[i1 * 3 + 2]);
                glm::vec3 v2(mesh.mVertices[i2 * 3], mesh.mVertices[i2 * 3 + 1], mesh.mVertices[i2 * 3 + 2]);

                // Möller–Trumboreアルゴリズムによるレイ-三角形交差判定
                glm::vec3 edge1 = v1 - v0;
                glm::vec3 edge2 = v2 - v0;
                glm::vec3 h = glm::cross(dir, edge2);
                float a = glm::dot(edge1, h);

                if (std::abs(a) < 1e-6)
                    continue;  // 平行またはほぼ平行

                float f = 1.0f / a;
                glm::vec3 s = origin - v0;
                float u = f * glm::dot(s, h);

                if (u < 0.0f || u > 1.0f)
                    continue;

                glm::vec3 q = glm::cross(s, edge1);
                float v = f * glm::dot(dir, q);

                if (v < 0.0f || u + v > 1.0f)
                    continue;

                float t = f * glm::dot(edge2, q);

                if (t > 1e-6) {
                    intersectionCount++;
                }
            }

            return intersectionCount;
        }

        // 内部ノードなら子ノードを再帰的に処理
        int leftCount = (node->left) ? rayMeshIntersect(origin, dir, mesh, node->left) : 0;
        int rightCount = (node->right) ? rayMeshIntersect(origin, dir, mesh, node->right) : 0;

        return leftCount + rightCount;
    }

    // 点がメッシュの内部にあるかどうかの判定
    bool isPointInside(const glm::vec3& point, const mCutMesh& mesh, BVHNode* bvh) {
        // 6方向にレイを飛ばして判定
        const glm::vec3 directions[6] = {
            glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0),
            glm::vec3(0, 1, 0), glm::vec3(0, -1, 0),
            glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)
        };

        int insideCount = 0;

        for (int i = 0; i < 6; i++) {
            int intersections = rayMeshIntersect(point, directions[i], mesh, bvh);
            if (intersections % 2 == 1) {
                insideCount++;
            }
        }

        // 過半数の方向でメッシュと交差する場合、内部にあると判定
        return insideCount > 3;
    }

    // BVHを使ったメッシュの内外判定と描画
    void drawsegment(ShaderProgram& shader, mCutMesh& mesh, mCutMesh& referenceObj, float alpha = 1.0f) {

        // 色の設定
        glm::vec4 insideColor = glm::vec4(0.2f, 0.8f, 0.2f, alpha);
        glm::vec4 outsideColor = glm::vec4(0.8f, 0.2f, 0.2f, alpha);

        // キャッシング用のID
        static GLuint lastMeshID = 0, lastRefID = 0;
        static BVHNode* refBVH = nullptr;
        static std::vector<GLuint> insideTriangles, outsideTriangles;

        // 処理対象が変わったらBVHを再構築
        if (lastMeshID != mesh.VAO || lastRefID != referenceObj.VAO) {
            // 前回のBVHを解放
            delete refBVH;


            // スケール係数（1.05倍に拡大）
            const float SCALE_FACTOR = 1.02f;

            // referenceObjをスケールアップした一時的なコピーを作成
            mCutMesh scaledRef = referenceObj;

            // 頂点を中心からスケールアップ
            glm::vec3 center(0.0f);
            for (size_t i = 0; i < referenceObj.mVertices.size(); i += 3) {
                center.x += referenceObj.mVertices[i];
                center.y += referenceObj.mVertices[i+1];
                center.z += referenceObj.mVertices[i+2];
            }
            center /= (referenceObj.mVertices.size() / 3);

            // 各頂点を中心からスケールアップ
            for (size_t i = 0; i < scaledRef.mVertices.size(); i += 3) {
                glm::vec3 dir(
                    scaledRef.mVertices[i] - center.x,
                    scaledRef.mVertices[i+1] - center.y,
                    scaledRef.mVertices[i+2] - center.z
                );

                scaledRef.mVertices[i] = center.x + dir.x * SCALE_FACTOR;
                scaledRef.mVertices[i+1] = center.y + dir.y * SCALE_FACTOR;
                scaledRef.mVertices[i+2] = center.z + dir.z * SCALE_FACTOR;
            }

            // 参照オブジェクトの全三角形インデックスを準備
            std::vector<uint32_t> allTriangles;
            for (uint32_t i = 0; i < scaledRef.mIndices.size() / 3; i++) {
                allTriangles.push_back(i);
            }
            // BVHを構築
            refBVH = buildBVH(scaledRef, allTriangles);

            // 内外判定を実行
            insideTriangles.clear();
            outsideTriangles.clear();

            for (uint32_t i = 0; i < mesh.mIndices.size() / 3; i++) {
                // 三角形の中心を計算
                glm::vec3 center(0.0f);
                for (int j = 0; j < 3; j++) {
                    uint32_t vertIdx = mesh.mIndices[i * 3 + j];
                    center += glm::vec3(mesh.mVertices[vertIdx * 3],
                                       mesh.mVertices[vertIdx * 3 + 1],
                                       mesh.mVertices[vertIdx * 3 + 2]);
                }
                center /= 3.0f;

                // 内外判定
                if (isPointInside(center, scaledRef, refBVH)) {
                    insideTriangles.push_back(i);
                } else {
                    outsideTriangles.push_back(i);
                }
            }

            lastMeshID = mesh.VAO;
            lastRefID = referenceObj.VAO;
        }

        // メッシュをバインド
        glBindVertexArray(mesh.VAO);

        // 内部の三角形を描画
        if (!insideTriangles.empty()) {
            // インデックスを展開
            std::vector<GLuint> indices;
            for (uint32_t triIdx : insideTriangles) {
                indices.push_back(mesh.mIndices[triIdx * 3]);
                indices.push_back(mesh.mIndices[triIdx * 3 + 1]);
                indices.push_back(mesh.mIndices[triIdx * 3 + 2]);
            }

            // 一時的なEBOを作成
            GLuint tempEBO;
            glGenBuffers(1, &tempEBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

            // 内部を緑色で描画
            shader.setUniform("vertColor", insideColor);
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

            // 一時的なEBOを解放
            glDeleteBuffers(1, &tempEBO);
        }

        // 外部の三角形を描画
        if (!outsideTriangles.empty()) {
            // インデックスを展開
            std::vector<GLuint> indices;
            for (uint32_t triIdx : outsideTriangles) {
                indices.push_back(mesh.mIndices[triIdx * 3]);
                indices.push_back(mesh.mIndices[triIdx * 3 + 1]);
                indices.push_back(mesh.mIndices[triIdx * 3 + 2]);
            }

            // 一時的なEBOを作成
            GLuint tempEBO;
            glGenBuffers(1, &tempEBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

            // 外部を赤色で描画
            shader.setUniform("vertColor", outsideColor);
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

            // 一時的なEBOを解放
            glDeleteBuffers(1, &tempEBO);
        }

        glBindVertexArray(0);
    }

    void drawsegment(ShaderProgram& shader, mCutMesh& mesh, mCutMesh& referenceObj, const glm::vec3& camPos, float alpha = 1.0f) {
        // 色の設定
        glm::vec4 insideColor = glm::vec4(0.2f, 0.8f, 0.2f, alpha);
        glm::vec4 outsideColor = glm::vec4(0.8f, 0.2f, 0.2f, alpha);

        // キャッシング用のID
        static GLuint lastMeshID = 0, lastRefID = 0;
        static BVHNode* refBVH = nullptr;
        static std::vector<GLuint> insideTriangles, outsideTriangles;

        // 処理対象が変わったらBVHを再構築
        if (lastMeshID != mesh.VAO || lastRefID != referenceObj.VAO) {
            // 前回のBVHを解放
            delete refBVH;

            // スケール係数（1.02倍に拡大）
            const float SCALE_FACTOR = 1.02f;

            // referenceObjをスケールアップした一時的なコピーを作成
            mCutMesh scaledRef = referenceObj;

            // 頂点を中心からスケールアップ
            glm::vec3 center(0.0f);
            for (size_t i = 0; i < referenceObj.mVertices.size(); i += 3) {
                center.x += referenceObj.mVertices[i];
                center.y += referenceObj.mVertices[i+1];
                center.z += referenceObj.mVertices[i+2];
            }
            center /= (referenceObj.mVertices.size() / 3);

            // 各頂点を中心からスケールアップ
            for (size_t i = 0; i < scaledRef.mVertices.size(); i += 3) {
                glm::vec3 dir(
                    scaledRef.mVertices[i] - center.x,
                    scaledRef.mVertices[i+1] - center.y,
                    scaledRef.mVertices[i+2] - center.z
                    );

                scaledRef.mVertices[i] = center.x + dir.x * SCALE_FACTOR;
                scaledRef.mVertices[i+1] = center.y + dir.y * SCALE_FACTOR;
                scaledRef.mVertices[i+2] = center.z + dir.z * SCALE_FACTOR;
            }

            // 参照オブジェクトの全三角形インデックスを準備
            std::vector<uint32_t> allTriangles;
            for (uint32_t i = 0; i < scaledRef.mIndices.size() / 3; i++) {
                allTriangles.push_back(i);
            }
            // BVHを構築
            refBVH = buildBVH(scaledRef, allTriangles);

            // 内外判定を実行
            insideTriangles.clear();
            outsideTriangles.clear();

            for (uint32_t i = 0; i < mesh.mIndices.size() / 3; i++) {
                // 三角形の中心を計算
                glm::vec3 center(0.0f);
                for (int j = 0; j < 3; j++) {
                    uint32_t vertIdx = mesh.mIndices[i * 3 + j];
                    center += glm::vec3(mesh.mVertices[vertIdx * 3],
                                        mesh.mVertices[vertIdx * 3 + 1],
                                        mesh.mVertices[vertIdx * 3 + 2]);
                }
                center /= 3.0f;

                // 内外判定
                if (isPointInside(center, scaledRef, refBVH)) {
                    insideTriangles.push_back(i);
                } else {
                    outsideTriangles.push_back(i);
                }
            }

            lastMeshID = mesh.VAO;
            lastRefID = referenceObj.VAO;
        }

        // 透明描画のための設定
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);

        // メッシュをバインド
        glBindVertexArray(mesh.VAO);

        // 三角形距離情報構造体
        struct TriangleDistanceInfo {
            bool isInside;          // 内部かどうか
            uint32_t triangleIndex; // 三角形インデックス
            float distance;         // カメラからの距離
        };

        std::vector<TriangleDistanceInfo> sortedTriangles;

        // 内部三角形の距離計算
        for (uint32_t triIdx : insideTriangles) {
            // 三角形の中心を計算
            glm::vec3 center(0.0f);
            for (int j = 0; j < 3; j++) {
                uint32_t vertIdx = mesh.mIndices[triIdx * 3 + j];
                center += glm::vec3(mesh.mVertices[vertIdx * 3],
                                    mesh.mVertices[vertIdx * 3 + 1],
                                    mesh.mVertices[vertIdx * 3 + 2]);
            }
            center /= 3.0f;

            // カメラからの距離
            float distance = glm::length(camPos - center);
            sortedTriangles.push_back({true, triIdx, distance});
        }

        // 外部三角形の距離計算
        for (uint32_t triIdx : outsideTriangles) {
            // 三角形の中心を計算
            glm::vec3 center(0.0f);
            for (int j = 0; j < 3; j++) {
                uint32_t vertIdx = mesh.mIndices[triIdx * 3 + j];
                center += glm::vec3(mesh.mVertices[vertIdx * 3],
                                    mesh.mVertices[vertIdx * 3 + 1],
                                    mesh.mVertices[vertIdx * 3 + 2]);
            }
            center /= 3.0f;

            // カメラからの距離
            float distance = glm::length(camPos - center);
            sortedTriangles.push_back({false, triIdx, distance});
        }

        // 遠いものから近いものへソート（透明度のための順序）
        std::sort(sortedTriangles.begin(), sortedTriangles.end(),
                  [](const TriangleDistanceInfo& a, const TriangleDistanceInfo& b) {
                      return a.distance > b.distance;
                  });

        // 一時的なEBO
        GLuint tempEBO;
        glGenBuffers(1, &tempEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);

        // 現在の色（初期化用）
        bool isCurrentColorInside = false;
        bool isColorInitialized = false;

        // ソートされた順序で三角形を描画
        for (const auto& tri : sortedTriangles) {
            // 内部/外部に応じて色を設定（色が変わる場合のみ）
            if (!isColorInitialized || tri.isInside != isCurrentColorInside) {
                shader.setUniform("vertColor", tri.isInside ? insideColor : outsideColor);
                isCurrentColorInside = tri.isInside;
                isColorInitialized = true;
            }

            // この三角形のインデックスを抽出
            std::vector<GLuint> indices = {
                mesh.mIndices[tri.triangleIndex * 3],
                mesh.mIndices[tri.triangleIndex * 3 + 1],
                mesh.mIndices[tri.triangleIndex * 3 + 2]
            };

            // EBOを更新して描画
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
            glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);
        }

        // 一時的なEBOを解放
        glDeleteBuffers(1, &tempEBO);

        // 元の設定に戻す
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glBindVertexArray(0);
    }

    // MCUTからメッシュを読み込む
    mCutMesh loadMeshFromFile(const char* filePath) {
        mCutMesh mesh;

        // ファイルを開く
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            return mesh;
        }

        // 一時的なデータ構造
        std::vector<glm::vec3> vertices;
        std::vector<std::vector<int>> faces;

        // OBJファイルを解析
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;

            if (type == "v") {
                // 頂点データ
                float x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(glm::vec3(x, y, z));
            }
            else if (type == "f") {
                // 面データ
                std::vector<int> face;
                std::string vertex;
                while (iss >> vertex) {
                    // 頂点インデックスを抽出（v/vt/vnの形式から最初の数字だけ）
                    size_t pos = vertex.find('/');
                    if (pos != std::string::npos) {
                        vertex = vertex.substr(0, pos);
                    }
                    // OBJの頂点インデックスは1始まりなので、0始まりに変換
                    face.push_back(std::stoi(vertex) - 1);
                }
                // 三角形の面だけを保存
                if (face.size() >= 3) {
                    // 多角形の三角形分割（ファンタイプ）
                    for (size_t i = 1; i < face.size() - 1; ++i) {
                        std::vector<int> triangle = {face[0], face[i], face[i + 1]};
                        faces.push_back(triangle);
                    }
                }
            }
        }
        file.close();

        // 頂点データをmCutMesh形式に変換
        mesh.mVertices.clear();
        mesh.mVertices.reserve(vertices.size() * 3);
        for (const auto& vertex : vertices) {
            mesh.mVertices.push_back(vertex.x);
            mesh.mVertices.push_back(vertex.y);
            mesh.mVertices.push_back(vertex.z);
        }

        // インデックスデータを変換
        mesh.mIndices.clear();
        mesh.mIndices.reserve(faces.size() * 3);
        for (const auto& face : faces) {
            for (int idx : face) {
                mesh.mIndices.push_back(static_cast<GLuint>(idx));
            }
        }

        // 面数の記録
        mesh.numFaces = faces.size();

        // デフォルトの色を設定
        mesh.mColor = glm::vec3(0.7f, 0.7f, 0.7f);

        return mesh;
    }

};

void draw_AllmCutMeshes(const std::vector<mCutMesh>& meshes, ShaderProgram& shader, const glm::vec3& camPos) {
    shader.use();

    // Enable transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    // Camera position
    glm::vec3 cameraPos = camPos;

    // Define mesh colors with alpha values for transparency

    std::vector<glm::vec4> meshColors = {
        glm::vec4(0.8f, 0.2f, 0.2f, 1.0f),  // Red (100% opacity)
        glm::vec4(0.9f, 0.6f, 0.6f, 0.9f),  // Pink (90% opacity)
        glm::vec4(0.2f, 0.8f, 0.8f, 0.9f),  // Cyan (90% opacity)
        glm::vec4(0.8f, 0.2f, 0.8f, 0.9f),  // Purple (90% opacity)
        glm::vec4(0.2f, 0.8f, 0.2f, 0.8f),  // Green (80% opacity)
        glm::vec4(0.8f, 0.8f, 0.2f, 0.8f),  // Yellow (80% opacity)
    };

    // Structure to store triangle information for sorting
    struct TriangleInfo {
        size_t meshIndex;       // Which mesh it belongs to
        uint32_t triangleIndex; // Triangle index within the mesh
        float distance;         // Distance from camera
    };

    std::vector<TriangleInfo> allTriangles;

    // Collect triangle information from all meshes
    for (size_t meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
        const auto& mesh = meshes[meshIdx];

        // Process each triangle in the mesh
        for (size_t triIdx = 0; triIdx < mesh.mIndices.size() / 3; triIdx++) {
            uint32_t idx1 = mesh.mIndices[triIdx * 3];
            uint32_t idx2 = mesh.mIndices[triIdx * 3 + 1];
            uint32_t idx3 = mesh.mIndices[triIdx * 3 + 2];

            // Get vertex positions
            glm::vec3 v1(mesh.mVertices[idx1 * 3], mesh.mVertices[idx1 * 3 + 1], mesh.mVertices[idx1 * 3 + 2]);
            glm::vec3 v2(mesh.mVertices[idx2 * 3], mesh.mVertices[idx2 * 3 + 1], mesh.mVertices[idx2 * 3 + 2]);
            glm::vec3 v3(mesh.mVertices[idx3 * 3], mesh.mVertices[idx3 * 3 + 1], mesh.mVertices[idx3 * 3 + 2]);

            // Calculate triangle center
            glm::vec3 center = (v1 + v2 + v3) / 3.0f;

            // Calculate distance from camera to triangle center
            float distance = glm::length(cameraPos - center);

            // Store triangle information
            allTriangles.push_back({meshIdx, static_cast<uint32_t>(triIdx), distance});
        }
    }

    // Sort triangles based on distance from camera (back-to-front for transparency)
    std::sort(allTriangles.begin(), allTriangles.end(),
              [](const TriangleInfo& a, const TriangleInfo& b) {
                  return a.distance > b.distance; // Far to near sorting
              });

    // Draw all triangles in sorted order
    GLuint lastVAO = -1;        // Track last VAO to minimize bindings
    glm::vec4 lastColor;        // Track last color to minimize shader uniform updates
    bool colorInitialized = false;

    // Create temporary EBO for each draw call
    GLuint tempEBO;
    glGenBuffers(1, &tempEBO);

    for (const auto& tri : allTriangles) {
        size_t meshIdx = tri.meshIndex;
        uint32_t triIdx = tri.triangleIndex;
        const auto& mesh = meshes[meshIdx];

        // Only switch VAO if needed
        if (lastVAO != mesh.VAO) {
            glBindVertexArray(mesh.VAO);
            lastVAO = mesh.VAO;
        }

        // Only update color if needed
        glm::vec4 currentColor = meshColors[meshIdx % meshColors.size()];
        if (!colorInitialized || lastColor != currentColor) {
            shader.setUniform("vertColor", currentColor);
            lastColor = currentColor;
            colorInitialized = true;
        }

        // Create indices for just this triangle
        std::vector<GLuint> indices = {
            mesh.mIndices[triIdx * 3],
            mesh.mIndices[triIdx * 3 + 1],
            mesh.mIndices[triIdx * 3 + 2]
        };

        // Bind and update temporary EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        // Draw the triangle
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);
    }

    // Clean up temporary EBO
    glDeleteBuffers(1, &tempEBO);

    // Restore original state
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}

void setUp(mCutMesh& srcMesh) {
    // エラーをクリア
    while (glGetError() != GL_NO_ERROR) {}

    // データの有効性チェック
    if (srcMesh.mVertices.empty() || srcMesh.mIndices.empty()) {
        std::cerr << "Error: Empty mesh data" << std::endl;
        return;
    }

    // Initialize vectors with the correct size
    std::vector<float> vertices(srcMesh.mVertices.size());
    std::vector<GLuint> indices(srcMesh.mIndices.size()); // GLuintに変更

    // Copy data from source mesh
    for(size_t i = 0; i < srcMesh.mVertices.size(); i++) {
        vertices[i] = srcMesh.mVertices[i];
    }

    for(size_t i = 0; i < srcMesh.mIndices.size(); i++) {
        indices[i] = static_cast<GLuint>(srcMesh.mIndices[i]); // 明示的キャスト
    }

    // インデックスの範囲チェック
    size_t vertexCount = vertices.size() / 3;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= vertexCount) {
            std::cerr << "Error: Index out of range at " << i << ": " << indices[i]
                      << " (vertex count: " << vertexCount << ")" << std::endl;
            return;
        }
    }

    // Create normals vector with the correct size
    std::vector<float> normals(vertices.size(), 0.0f);

    for (size_t i = 0; i < indices.size(); i += 3) {
        // 三角形の頂点インデックス
        GLuint i0 = indices[i];
        GLuint i1 = indices[i + 1];
        GLuint i2 = indices[i + 2];

        // 三角形の頂点座標
        glm::vec3 v0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        glm::vec3 v2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        // エッジベクトル
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        // 法線計算（外積を正規化）
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

        // 各頂点に法線を加算（後で正規化）
        for (GLuint idx : {i0, i1, i2}) {
            normals[idx * 3]     += normal.x;
            normals[idx * 3 + 1] += normal.y;
            normals[idx * 3 + 2] += normal.z;
        }
    }

    // 頂点ごとの法線を正規化
    for (size_t i = 0; i < normals.size(); i += 3) {
        glm::vec3 n(normals[i], normals[i + 1], normals[i + 2]);
        float length = glm::length(n);
        if (length > 0.0001f) { // 0除算を避けるためのチェック
            n /= length;
        } else {
            n = glm::vec3(0.0f, 1.0f, 0.0f); // デフォルト法線
        }
        normals[i]     = n.x;
        normals[i + 1] = n.y;
        normals[i + 2] = n.z;
    }

    // バッファの生成
    glGenVertexArrays(1, &srcMesh.VAO);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenVertexArrays: " << err << std::endl;
        return;
    }

    glGenBuffers(1, &srcMesh.VBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenBuffers(VBO): " << err << std::endl;
        return;
    }

    glGenBuffers(1, &srcMesh.EBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenBuffers(EBO): " << err << std::endl;
        return;
    }

    glGenBuffers(1, &srcMesh.NBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenBuffers(NBO): " << err << std::endl;
        return;
    }

    glBindVertexArray(srcMesh.VAO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindVertexArray: " << err << std::endl;
        return;
    }

    // 頂点バッファ
    glBindBuffer(GL_ARRAY_BUFFER, srcMesh.VBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindBuffer(VBO): " << err << std::endl;
        return;
    }

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBufferData(VBO): " << err << std::endl;
        return;
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glVertexAttribPointer(pos): " << err << std::endl;
        return;
    }

    glEnableVertexAttribArray(0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glEnableVertexAttribArray(pos): " << err << std::endl;
        return;
    }

    // 法線バッファ
    glBindBuffer(GL_ARRAY_BUFFER, srcMesh.NBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindBuffer(NBO): " << err << std::endl;
        return;
    }

    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(GLfloat), normals.data(), GL_STATIC_DRAW);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBufferData(NBO): " << err << std::endl;
        return;
    }

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glVertexAttribPointer(normal): " << err << std::endl;
        return;
    }

    glEnableVertexAttribArray(1);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glEnableVertexAttribArray(normal): " << err << std::endl;
        return;
    }

    // インデックスバッファ
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, srcMesh.EBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindBuffer(EBO): " << err << std::endl;
        return;
    }

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBufferData(EBO): " << err << std::endl;
        return;
    }

    glBindVertexArray(0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindVertexArray(0): " << err << std::endl;
    }
}

mCutMesh *cutterMesh;

int main()
{

    if (!initOpenGL())
        {
            // An error occured
            std::cerr << "GLFW initialization failed" << std::endl;
            return -1;
        }

    ShaderProgram shaderProgram;
    shaderProgram.loadShaders("../../../shaders/basic.vert", "../../../shaders/basic.frag");

    // メッシュの読み込み
    const char* cutMeshFilePath = "../../../data/liver.obj";
    cutterMesh = new mCutMesh(cutterMesh->loadMeshFromFile(cutMeshFilePath));
    cutterMesh->mColor = glm::vec3(0.2f, 0.8f, 0.2f);
    setUp(*cutterMesh);


    while (!glfwWindowShouldClose(gWindow))
    {
        showFPS(gWindow);
        // Poll for and process events
        glfwPollEvents();
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ////////////////////////////////////////
        model = glm::translate(glm::mat4(1.0f), objPos);
        cutterMesh->draw(shaderProgram, 0.7f);

        OrbitCam.UpdateOrbit();

        glfwSwapBuffers(gWindow);
    }
    glfwTerminate();

    return 0;
}

bool initOpenGL()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight, "Window", NULL, NULL);
    glfwMakeContextCurrent(gWindow);
    glewExperimental = GL_TRUE;
    glewInit();

    glfwSetKeyCallback(gWindow, glfw_onKey);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);
    glfwSetFramebufferSizeCallback(gWindow, glfw_OnFramebufferSize);
    glfwSetCursorPosCallback(gWindow, glfw_onMouseMoveOrbit);
    glfwSetScrollCallback(gWindow, glfw_onMouseScroll);

    // Hides and grabs cursor, unlimited movement
    //glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPos(gWindow, gWindowWidth / 2.0, gWindowHeight / 2.0);

    glClearColor(0.23f, 0.38f, 0.47f, 1.0f);

    glViewport(0, 0, gWindowWidth, gWindowHeight);
    glEnable(GL_DEPTH_TEST);

    return true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed = true;
        // スクリーン座標をそのまま渡す
        std::cout << "triMesh Find Hit" << std::endl;
        FindHit(xpos, ypos, cutterMesh->mVertices, cutterMesh->mIndices);
        }
    else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed = false;
        hit_index = -1;
        std::cout << "hit_index: "<<  hit_index << std::endl;
    }

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
        mousePressed = true;
        // スクリーン座標をそのまま渡す
        std::cout << "triMesh Find Hit" << std::endl;
        FindHit(xpos, ypos, cutterMesh->mVertices, cutterMesh->mIndices);
    }
    else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
        mousePressed = false;
        hit_index = -1;
        std::cout << "hit_index: "<<  hit_index << std::endl;
    }
  }

void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_F1:
            gWireframe = !gWireframe;
            if (gWireframe)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            else
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            break;
        case GLFW_KEY_C: // enter/exit cutting mode
            cutting = !cutting;
            break;
        case GLFW_KEY_D: // enter/exit cutting mode
             cutting = false;
             cutMode = false;
             deformMode = true;
             deformModeInit = true;
             std::cout << "cutting: " << cutting << std::endl;
             std::cout << "cutMode: " << cutMode << std::endl;
             std::cout << "deformMode: " << deformMode << std::endl;
             break;
         case GLFW_KEY_R: // remove cutting triangle mesh
             meshcut = true;
             break;
         case GLFW_KEY_A: // remove cutting triangle mesh
             restore = true;
             break;
         case GLFW_KEY_M: // remove cutting triangle mesh
             break;
         case GLFW_KEY_P: // remove cutting triangle mesh
             break;
         case GLFW_KEY_V: // remove cutting triangle mesh
             break;
       }
    }

void glfw_OnFramebufferSize(GLFWwindow* window, int width, int height)
{
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void glfw_onMouseMoveOrbit(GLFWwindow* window, double posX, double posY) {

    static glm::vec2 lastMousePos = glm::vec2(0, 0);

    // 変数変更後にカメラ関連ベクトルを更新する関数
    auto updateCameraVectors = []() {
        OrbitCam.cameraDirection = glm::normalize(OrbitCam.cameraPos - OrbitCam.cameraTarget);
        OrbitCam.cameraRight = glm::normalize(glm::cross(OrbitCam.up, OrbitCam.cameraDirection));
        OrbitCam.cameraUp = glm::normalize(glm::cross(OrbitCam.cameraDirection, OrbitCam.cameraRight));
    };

    if (isDragging == false)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == 1)
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != 1) {
                OrbitCam.gYaw -= ((float)posX - lastMousePos.x) * OrbitCam.MOUSE_SENSITIVITY;
                OrbitCam.gPitch += ((float)posY - lastMousePos.y) * OrbitCam.MOUSE_SENSITIVITY;
                updateCameraVectors();
            }


    if (isDragging == true)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS &&
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {

            updateCameraVectors();
            // 画面空間での移動ベクトルを取得

            float dx = ((float)posX - lastMousePos.x) * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
            float dy = (lastMousePos.y - (float)posY) * OrbitCam.LIGHT_MOUSE_SENSITIVITY;

            // 現在のカメラ方向に基づいて世界空間の移動ベクトルに変換
            glm::vec3 moveDirection = OrbitCam.cameraRight * dx + OrbitCam.cameraUp * dy;

            // 全ての頂点に移動を適用
            for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                cutterMesh->mVertices[i]     += moveDirection.x;
                cutterMesh->mVertices[i + 1] += moveDirection.y;
                cutterMesh->mVertices[i + 2] += moveDirection.z;
            }
            setUp(*cutterMesh);  // 変更を適用
        }

    if (isDragging == false)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS &&
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            glm::vec3 moveRight = OrbitCam.cameraRight * ((float)posX - lastMousePos.x) * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
            glm::vec3 moveUp = OrbitCam.cameraUp * (lastMousePos.y - (float)posY) * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
            OrbitCam.cameraTarget.x  -= moveRight.x;
            OrbitCam.cameraTarget.y  -= moveUp.y;
            OrbitCam.cameraTarget.z  -= moveRight.z + moveUp.z;  // Z 軸は変更なし（必要なら適宜変更）
            updateCameraVectors();
        }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == 1)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == 1) {
            updateCameraVectors();
            // cutterMesh をライトと同じように移動
            for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                cutterMesh->mVertices[i]     += (OrbitCam.cameraDirection * ((float)posY - lastMousePos.y) * OrbitCam.LIGHT_MOUSE_SENSITIVITY).x;
                cutterMesh->mVertices[i + 1] += (OrbitCam.cameraDirection * ((float)posY - lastMousePos.y) * OrbitCam.LIGHT_MOUSE_SENSITIVITY).y;
                cutterMesh->mVertices[i + 2] += (OrbitCam.cameraDirection * ((float)posY - lastMousePos.y) * OrbitCam.LIGHT_MOUSE_SENSITIVITY).z;
            }
            setUp(*cutterMesh);  // 変更を適用
        }

    if (isDragging == true)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == 1)
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != 1)
            // Inside the if block for cutMode==true and left mouse button pressed
            {
                // Calculate rotation based on mouse movement
                float rotX = ((float)posY - lastMousePos.y) * 0.01f; // Adjust sensitivity as needed
                float rotY = ((float)posX - lastMousePos.x) * 0.01f;

                // Get the center of the cutter mesh (for rotation around its center)
                glm::vec3 center(0.0f);
                for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                    center.x += cutterMesh->mVertices[i];
                    center.y += cutterMesh->mVertices[i + 1];
                    center.z += cutterMesh->mVertices[i + 2];
                }
                center /= (cutterMesh->mVertices.size() / 3);

                // Create rotation matrices
                glm::mat4 transform = glm::mat4(1.0f);
                transform = glm::translate(transform, center);
                transform = glm::rotate(transform, rotX, OrbitCam.cameraRight); // Rotate around camera right vector
                transform = glm::rotate(transform, rotY, OrbitCam.cameraUp);    // Rotate around camera up vector
                transform = glm::translate(transform, -center);

                // Apply rotation to all vertices
                for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                    glm::vec4 vertex(
                        cutterMesh->mVertices[i],
                        cutterMesh->mVertices[i + 1],
                        cutterMesh->mVertices[i + 2],
                        1.0f
                    );

                    // Apply transformation
                    vertex = transform * vertex;

                    // Store transformed vertex
                    cutterMesh->mVertices[i]     = vertex.x;
                    cutterMesh->mVertices[i + 1] = vertex.y;
                    cutterMesh->mVertices[i + 2] = vertex.z;
                }

                // Apply changes to the mesh
                setUp(*cutterMesh);
            }
    lastMousePos.x = (float)posX;
    lastMousePos.y = (float)posY;
}

float scaleSpeed = 1.1;

void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY)
{
    double fov = OrbitCam.gFOV + deltaY * OrbitCam.ZOOM_SENSITIVITY;
    fov = glm::clamp(fov, 1.0, 120.0);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != 1)
        OrbitCam.gFOV = fov;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == 1)
        {
           if (deltaY > 0) {
              for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                  cutterMesh->mVertices[i] *= scaleSpeed;
                  cutterMesh->mVertices[i+1] *= scaleSpeed;
                  cutterMesh->mVertices[i+2] *= scaleSpeed;
              }
          } else if (deltaY < 0) {
               for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                  cutterMesh->mVertices[i] /= scaleSpeed;
                  cutterMesh->mVertices[i+1] /= scaleSpeed;
                  cutterMesh->mVertices[i+2] /= scaleSpeed;
              }
          }
           setUp(*cutterMesh);
        }
}

void showFPS(GLFWwindow* window)
{
    static double previousSeconds = 0.0;
    static int frameCount = 0;
    double elapsedSeconds;
    double currentSeconds = glfwGetTime(); // returns number of seconds since GLFW started, as double float

    elapsedSeconds = currentSeconds - previousSeconds;

    // Limit text updates to 4 times per second
    if (elapsedSeconds > 0.25)
    {
        previousSeconds = currentSeconds;
        double fps = (double)frameCount / elapsedSeconds;
        double msPerFrame = 1000.0 / fps;

        // The C++ way of setting the window title
        std::ostringstream outs;
        outs.precision(3);	// decimal places
        outs << std::fixed
            << "APP" << "    "
            << "FPS: " << fps << "    "
            << "Frame Time: " << msPerFrame << " (ms)";
        glfwSetWindowTitle(window, outs.str().c_str());

        // Reset for next average.
        frameCount = 0;
    }

    frameCount++;
}
