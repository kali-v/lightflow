#include <lightflow/tensor.h>

class FashionMnistDataset {
  private:
    int limit;

  public:
    std::vector<Tensor> x;
    std::vector<Tensor> y;

    FashionMnistDataset(const char* xfile, const char* yfile, int limit = -1);

    std::vector<Tensor> get_item(int id);

    int len();
};
