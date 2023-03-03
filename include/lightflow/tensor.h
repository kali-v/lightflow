#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <string>
#include <vector>

typedef std::vector<float> Vec1D;
typedef std::vector<std::vector<float>> Vec2D;
typedef std::vector<int> DimVec;
typedef std::function<float(float, float)> OperationFunc;
typedef std::function<void(float*, float*, float*, int, int, int)> MatmulFunc;

inline float add(float a, float b) { return a + b; }

inline float sub(float a, float b) { return a - b; }

inline float mul(float a, float b) { return a * b; }

inline float ddiv(float a, float b) { return a / b; }

enum class Device { CPU = 0, CUDA = 1 };
const Device defdev = getenv("LF_DEFDEV") ? static_cast<Device>(atoi(getenv("LF_DEFDEV"))) : Device::CPU;

class Tensor {
  private:
    std::size_t size();
    void check_same_shape(Tensor& other, bool allow_scalar = true);
    DimVec get_short_shape();

    void _matmul_deep(Tensor& other, float* res, MatmulFunc matmul_fn);

  public:
    Vec1D data;
    std::vector<int> shape;
    DimVec dshape;
    Device device;

    bool require_grad;
    Tensor* grad = nullptr;

    std::function<void()> backward_fn;
    std::vector<Tensor*> children;

    Tensor(const std::vector<int>& shape, bool require_grad = false, Device device = defdev);

    Tensor(const std::vector<int>& shape, const Vec1D tensor, std::vector<Tensor*> children = {},
           bool require_grad = false, Device device = defdev);

    Tensor(const std::vector<int>& shape, const Vec2D tensor, std::vector<Tensor*> children = {},
           bool require_grad = false, Device device = defdev);

    Tensor(const std::vector<int>& shape, const float constant, std::vector<Tensor*> children = {},
           bool require_grad = false, Device device = defdev);

    ~Tensor();

    Tensor apply(std::function<float(float)> function);
    Tensor apply_operator(Tensor& other, OperationFunc operation_fn);

    Vec1D get_row(int row_num);
    Vec1D get_col(int row_num);
    Tensor get_block(int n);
    Tensor get_channel(int channel);
    void add_channel(Tensor& channel);

    static DimVec normalize_shape(DimVec shape);

    static Tensor scalar(float value);
    static Tensor scalar(int value);

    static Tensor random(DimVec shape, float from = 0, float to = 1);

    static Tensor ones(const std::vector<int> shape);
    static Tensor zeros(const std::vector<int> shape);

    Tensor operator+(Tensor& other);
    Tensor operator-(Tensor& other);
    Tensor operator*(Tensor& other);
    Tensor operator/(Tensor& other);

    Tensor operator+(float value);
    Tensor operator-(float value);
    Tensor operator*(float value);
    Tensor operator/(float value);

    Tensor pow(Tensor& other);
    Tensor pow(float other);
    Tensor sqrt();

    Tensor reshape(DimVec new_shape);
    Tensor transpose();
    Tensor rot180();
    Tensor pad(DimVec padding, float value);

    Tensor matmul(Tensor& other);
    Tensor channelwise_sum(Tensor& other);
    Tensor correlate(Tensor& filter, DimVec stride = {1, 1, 1}, DimVec padding = {0, 0, 0});

    int argmax();

    float max();
    float min();
    float sum();

    void operator-=(Tensor other);
    void operator+=(Tensor other);
    bool operator==(Tensor& other);

    void fill(float value);
    void fill(Vec1D data);
    void fill(Vec2D data);

    void add_grad(Vec1D grad);
    void set_grad(Vec1D grad);

    void set_grad(Tensor* grad);

    bool has_same_shape(Tensor& other);

    Tensor to(Device device);

    void backward();

    std::string to_string();
};

#endif
