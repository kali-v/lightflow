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

void check_cpu(const char* fc_name, const Device device);

class Tensor {
  private:
    void check_same_shape(Tensor& other, bool allow_scalar = true) const;
    DimVec get_short_shape() const;

    void _matmul_deep(Tensor& other, float* res, MatmulFunc matmul_fn);

  public:
    std::size_t size() const;
    Vec1D data_;
    DimVec shape_;
    DimVec dshape_;
    Device device_;

    bool requires_grad_;
    Tensor* grad_ = nullptr;

    std::function<void()> backward_fn_;
    std::vector<Tensor*> children_;

    float* cu_data_ = nullptr;

    Tensor(const DimVec& shape, bool requires_grad = false, Device device = defdev);

    Tensor(const DimVec& shape, const Vec1D& tensor, std::vector<Tensor*> children = {}, bool requires_grad = false,
           Device device = defdev);

    Tensor(const DimVec& shape, const Vec2D& tensor, std::vector<Tensor*> children = {}, bool requires_grad = false,
           Device device = defdev);

    Tensor(const DimVec& shape, const float constant, std::vector<Tensor*> children = {}, bool requires_grad = false,
           Device device = defdev);

    ~Tensor();

    Tensor apply(std::function<float(float)> function);
    Tensor apply_operator(Tensor& other, OperationFunc operation_fn);

    Tensor get_block(int n);
    Tensor get_channel(int channel);
    void add_channel(const Tensor& channel);

    static DimVec normalize_shape(const DimVec shape);

    static Tensor& scalar(float value);
    static Tensor& scalar(int value);

    static Tensor random(DimVec shape, float from = 0, float to = 1);

    static Tensor ones(const DimVec shape);
    static Tensor zeros(const DimVec shape);

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
    Tensor log();

    Tensor reshape(DimVec new_shape);
    Tensor transpose();
    Tensor rot180();
    Tensor pad(DimVec padding, float value);

    Tensor matmul(Tensor& other);
    Tensor channelwise_sum(Tensor& other);
    Tensor correlate(Tensor& filter, DimVec stride = {1, 1}, DimVec padding = {0, 0});

    int argmax();

    float max();
    float min();
    float sum();

    void operator-=(Tensor other);
    void operator+=(Tensor other);
    bool operator==(Tensor& other);

    void fill(float value);
    void fill(const Vec1D& data);
    void fill(const Vec2D& data);

    void add_grad(Vec1D grad);
    void add_grad(const Tensor grad);

    void set_grad(Vec1D grad);
    void set_grad(Tensor* grad);

    bool has_same_shape(Tensor& other);

    Tensor to(Device device);

    void backward();

    std::string to_string();
};

typedef std::function<Tensor(Tensor&, Tensor&, DimVec stride, DimVec padding)> CorrelateFunc;

#endif
