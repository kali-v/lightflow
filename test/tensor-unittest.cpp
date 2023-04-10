#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include "gtest/gtest.h"

#include <lightflow/loss.h>
#include <lightflow/nn.h>
#include <lightflow/tensor.h>

using namespace std;

bool are_same_tensors(Vec1D a, Vec1D b) { return std::equal(a.begin(), a.end(), b.begin()); }

bool compare_tensors(Tensor exp, Tensor com, int i = 0) {
    bool result = exp == com;

    if (!result) {
        std::cout << std::to_string(i) << " tensor; expected:\n"
                  << exp.to_string() << "; got:\n" + com.to_string() << std::endl;
    }

    return result;
}

Vec2D load_test_data() {
    std::string test_name = std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::transform(test_name.begin(), test_name.end(), test_name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::ifstream fin("test/data/" + test_name + ".dat");
    if (!fin) {
        std::cout << "Test file for " << test_name << " not found." << std::endl;
        raise(SIGTERM);
    }

    Vec2D data;
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream ss(line);
        data.emplace_back(std::istream_iterator<float>(ss), std::istream_iterator<float>());
    }

    return data;
}

TEST(TENSOR_OPERATIONS, Add) {
    vector<vector<float>> examples = {{2.0, 3.0, 4.0, 5.0}, {2.0, 3.0, 4.0, 5.0}, {-15.4}, {5.0}, {0}, {0}};

    vector<vector<float>> results = {{4, 6, 8, 10}, {-10.4}, {0}};

    for (int i = 0; i < examples.size(); i += 2) {
        const std::vector<int> tsize = {1, (int)examples[i].size()};

        Tensor a = Tensor(tsize, (vector<float>)examples[i]);
        Tensor b = Tensor(tsize, (vector<float>)examples[i + 1]);

        Tensor computed = a + b;
        Tensor expected = Tensor(tsize, results[(int)i / 2]);

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Sub) {
    typedef vector<vector<vector<float>>> TestData;

    TestData examples = {{{2.0, 3.0}, {4.0, 5.0}}, {{2.0, 1.4}, {3.64, 6.2}}};

    TestData results = {{{0, 1.6}, {0.36, -1.2}}};

    for (int i = 0; i < examples.size(); i += 2) {
        const std::vector<int> tsize = {2, 2};

        Tensor a = Tensor(tsize, examples[i]);
        Tensor b = Tensor(tsize, examples[i + 1]);

        Tensor computed = a - b;
        Tensor expected = Tensor(tsize, results[(int)i / 2]);

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Mul) {
    vector<vector<float>> examples = {{2.0, 3.0, 4.0, 5.0}, {2.0, 3.0, 4.0, 5.0}, {-15.4, 4.6}, {5.0}, {0}, {0}};

    vector<vector<float>> results = {{4, 9, 16, 25}, {-77, 23}, {0}};

    for (int i = 0; i < examples.size(); i += 2) {
        const int max_tsize = std::max((int)examples[i].size(), (int)examples[i + 1].size());

        Tensor a = Tensor({(int)examples[i].size()}, (vector<float>)examples[i]);
        Tensor b = Tensor({(int)examples[i + 1].size()}, (vector<float>)examples[i + 1]);

        Tensor computed = a * b;
        Tensor expected = Tensor({max_tsize}, results[(int)i / 2]);

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Div) {
    vector<vector<float>> examples = {
        {2.0, 3.0, 4.0, 6.0}, {2.0, 3.0, -4.0, 5.0}, {-15.4, 4.6}, {5.0}, {23.21343}, {1.3123}};

    vector<vector<float>> results = {{1, 1, -1, 1.2}, {-3.08, 0.92}, {17.6891}};

    for (int i = 0; i < examples.size(); i += 2) {
        const int max_tsize = std::max((int)examples[i].size(), (int)examples[i + 1].size());

        Tensor a = Tensor({(int)examples[i].size()}, (vector<float>)examples[i]);
        Tensor b = Tensor({(int)examples[i + 1].size()}, (vector<float>)examples[i + 1]);

        Tensor computed = a / b;
        Tensor expected = Tensor({max_tsize}, results[(int)i / 2]);

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Matmul) {
    vector<Tensor> examples = {Tensor({1, 2}, {3.45, 4.321}),
                               Tensor({2, 1}, {83.54, 4.2341}),
                               Tensor({2, 2}, {6.23, 1.42, 65.22, 6.23}),
                               Tensor({2, 2}, {8.45, 1.42, 13.43, 8.97}),
                               Tensor({3, 2}, {8.45, -1.42, -13.43, 8.97, 1.23, 4.64}),
                               Tensor({2, 3}, {2, -3.44, 3.11, -8.97, 8.32, -1.11})};

    vector<Tensor> results = {
        Tensor::scalar(306.5085f), Tensor({2, 2}, {71.7141, 21.5840, 634.7779, 148.4955}),
        Tensor({3, 3}, {29.6374, -40.8824, 27.8557, -107.3209, 120.8296, -51.7240, -39.1608, 34.3736, -1.3251})};

    for (int i = 0; i < examples.size(); i += 2) {
        Tensor a = examples[i];
        Tensor b = examples[i + 1];
        Tensor computed = a.matmul(b);

        Tensor expected = results[i / 2];

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Correlate) {
    vector<Tensor> examples = {
        Tensor({1, 5}, {1, 2, 4, 5, 6}),
        Tensor({1, 2}, {1, 2}),
        Tensor({1, 5}, {1, 2, 4, 5, 6}),
        Tensor({1, 2}, {1, 2}),
        Tensor({2, 5}, {1, 2, 4, 5, 6, 1, 2, 4, 5, 6}),
        Tensor({2, 2}, {2, 3, 5, 8}),
        Tensor({1, 4}, {2, 3, 4, 6}),
        Tensor({1, 3}, {3, 2, 2}),
        Tensor({4, 4}, {0.2707, 0.7157, 0.7806, 0.8339, 0.4473, 0.4246, 0.7293, 0.1266, 0.5947, 0.5582, 0.1281, 0.8958,
                        0.3792, 0.4531, 0.1841, 0.0365}),
        Tensor({2, 2}, {2, 4.2, 3.8, 6.1}),
        Tensor({4, 5}, {0.2707, 0.7157, 0.7806, 0.8339, 1, 0.4473, 0.4246, 0.7293, 0.1266, 1,
                        0.5947, 0.5582, 0.1281, 0.8958, 1, 0.3792, 0.4531, 0.1841, 0.0365, 1}),
        Tensor({2, 2}, {2, 4.2, 3.8, 6.1}),
    };
    DimVec strides[] = {{1, 1}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, {2, 2}};
    vector<int> paddings[] = {{0, 0}, {0, 0}, {0, 0}, {0, 2}, {0, 0}, {0, 0}};

    vector<Tensor> results = {Tensor({1, 4}, {5., 10., 14., 17.}),
                              Tensor({1, 2}, {5., 14.}),
                              Tensor({1, 2}, {29., 83.}),
                              Tensor({1, 6}, {4, 10, 20, 29, 24, 18}),
                              Tensor({2, 2}, {7.8371, 8.6072, 7.7387, 4.9408}),
                              Tensor({2, 2}, {7.8371, 8.6072, 7.7387, 4.9408})};

    for (int i = 0; i < examples.size(); i += 2) {
        Tensor a = examples[i];
        Tensor b = examples[i + 1];

        Tensor computed = a.correlate(b, strides[i / 2], paddings[i / 2]);

        Tensor expected = results[i / 2];

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Rotate180) {
    vector<Tensor> examples = {Tensor({1, 3}, {1, 2, 3}), Tensor({2, 3}, {1, 2, 3, 4, 5, 6}),
                               Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})};

    vector<Tensor> results = {Tensor({1, 3}, {3, 2, 1}), Tensor({2, 3}, {6, 5, 4, 3, 2, 1}),
                              Tensor({3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1})};

    for (int i = 0; i < examples.size(); i++) {
        Tensor a = examples[i];
        Tensor computed = a.rot180();

        Tensor expected = results[i];

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(TENSOR_OPERATIONS, Pad_1) {
    Tensor example = Tensor({1, 1, 2, 2}, {1, 2, 3, 4});
    Tensor result = Tensor({1, 1, 4, 4}, {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0});

    Tensor computed = example.pad({1, 1}, 0.0);

    ASSERT_TRUE(compare_tensors(result, computed));
}

TEST(TENSOR_OPERATIONS, Pad_2) {
    Vec2D data = load_test_data();
    Tensor example = Tensor({3, 2, 7, 4}, data[0]);
    Tensor result = Tensor({3, 2, 9, 8}, data[1]);

    Tensor computed = example.pad({1, 2}, 0.0);

    ASSERT_TRUE(compare_tensors(result, computed));
}

TEST(TENSOR_OPERATIONS, Pad_3) {
    Tensor example = Tensor({1, 1, 2, 2}, {1, 2, 3, 4});

    Tensor computed = example.pad({0, 0}, 0);

    ASSERT_TRUE(compare_tensors(example, computed));
}

/*
 * GRADIENT TESTS
 */

TEST(GRAD, Add_1) {
    Tensor a = Tensor({1}, 1.0f, {}, true);
    Tensor b = Tensor({1}, 2.0f, {}, true);
    Tensor c = a + b;

    c.grad_ = new Tensor(c.shape_, 1.0f);
    c.backward();

    ASSERT_TRUE(compare_tensors(Tensor({1}, 1.0f), *(a.grad_)));
    ASSERT_TRUE(compare_tensors(Tensor({1}, 1.0f), *(b.grad_)));
    ASSERT_TRUE(compare_tensors(Tensor({1}, 1.0f), *(c.grad_)));
}

TEST(GRAD, Operations_1) {
    Tensor a = Tensor({1}, 5, {}, true);
    Tensor b = Tensor({1}, 3.1, {}, true);
    Tensor c = Tensor({1}, 2, {}, true);
    Tensor d = Tensor({1}, 3.2, {}, true);
    Tensor e = Tensor({1}, 11.86, {}, true);
    Tensor f = Tensor({1}, 1.23, {}, true);

    Tensor ab = a * b;
    Tensor cc = ab.pow(c);
    Tensor dd = cc / d;
    Tensor ee = dd - e;
    Tensor ff = ee + f;

    ff.grad_ = new Tensor(ff.shape_, 1.0f);
    ff.backward();

    std::vector<Tensor> exp_tensors = {
        Tensor({1}, 30.03125, {}, true), Tensor({1}, 48.4375, {}, true), Tensor({1}, 205.77713, {}, true),
        Tensor({1}, -23.4619, {}, true), Tensor({1}, -1, {}, true),      Tensor({1}, 1, {}, true),
    };

    std::vector<Tensor> com_tensors = {*a.grad_, *b.grad_, *c.grad_, *d.grad_, *e.grad_, *f.grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, Padding_1) {
    Vec2D data = load_test_data();
    Tensor* a = new Tensor({2, 3, 2, 2}, data[0], {}, true);

    Padding p = Padding({2, 2}, 0.0);
    Tensor* c = p.forward(a);

    c->grad_->fill(data[1]);
    c->backward();

    Tensor exp = Tensor({2, 3, 2, 2}, data[2]);

    ASSERT_TRUE(compare_tensors(exp, *a->grad_));
}

TEST(GRAD, MaxPool_1) {
    Tensor* a = new Tensor({2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 10, 1, 9, 9}, {}, true);

    MaxPool2D mp = MaxPool2D(2);
    Tensor* c = mp.forward(a);

    c->grad_->fill({1, 2, 3, 4});
    c->backward();

    Tensor exp = Tensor({2, 2, 2, 2}, {0, 0, 0, 1, 0, 0, 0, 2, 3, 0, 0, 0, 4, 0, 0, 0});

    ASSERT_TRUE(compare_tensors(exp, *a->grad_));
}

TEST(GRAD, MaxPool_2) {
    Tensor* a = new Tensor(
        {1, 3, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, 10, 1, 9, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1}, {}, true);

    MaxPool2D mp = MaxPool2D(2);
    Tensor* c = mp.forward(a);

    c->grad_->fill({1, 2, 3});
    c->backward();

    Tensor exp = Tensor({1, 3, 3, 3}, {0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 2.,
                                       0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0.});

    ASSERT_TRUE(compare_tensors(exp, *a->grad_));
}

TEST(GRAD, Loss_1) {
    Tensor a = Tensor({1}, 5, {}, true);
    Tensor b = Tensor({1}, 3.1, {}, true);
    Tensor c = Tensor({1}, 6, {}, true);

    Tensor l = l2_loss(new Tensor(a + b), &c);
    l.backward();

    std::vector<Tensor> exp_tensors = {
        Tensor::scalar((float)4.2),
        Tensor::scalar((float)4.2),
    };

    std::vector<Tensor> com_tensors = {*a.grad_, *b.grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, Loss_2) {
    Tensor a = Tensor({2, 1}, {5, 3}, {}, true);
    Tensor b = Tensor({2, 1}, {4, 3}, {}, true);
    Tensor c = Tensor({2, 1}, {10, 6}, {}, true);

    Tensor l = l2_loss(new Tensor(a + b), &c);
    l.backward();

    std::vector<Tensor> exp_tensors = {
        Tensor({2, 1}, {-2, 0}),
        Tensor({2, 1}, {-2, 0}),
    };

    std::vector<Tensor> com_tensors = {*a.grad_, *b.grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, Loss_3) {
    Tensor a = Tensor({2, 2}, {{1.3, -3.6}, {2.3, 3.2}}, {}, true);
    Tensor b = Tensor({2, 2}, {{1.2, 8.53}, {1.45, 2.3}}, {}, true);
    Tensor c = Tensor({2, 2}, {{2.5, -25.3}, {3.56, 3.11}}, {}, true);

    Tensor l = l2_loss(new Tensor(a * b), &c);
    l.backward();

    std::vector<Tensor> exp_tensors = {Tensor({2, 2}, {{-2.2560, -92.2605}, {-0.6525, 19.5500}}),
                                       Tensor({2, 2}, {{-2.4440, 38.9376}, {-1.0350, 27.2000}})};

    std::vector<Tensor> com_tensors = {*a.grad_, *b.grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, Loss_4) {
    Tensor a = Tensor({2, 2}, {{1.3, -3.6}, {2.3, 3.2}}, {}, true);
    Tensor b = Tensor({2, 2}, {{1.2, 8.53}, {1.45, 2.3}}, {}, true);
    Tensor c = Tensor({2, 2}, {{-4.2, 2.3}, {3.56, 23.11}}, {}, true);

    Tensor q = a.matmul(b);

    Tensor l = l2_loss(&q, &c);
    l.backward();

    std::vector<Tensor> exp_tensors = {Tensor({2, 2}, {{9.9795, 3.9074}, {75.2211, 28.9334}}),
                                       Tensor({2, 2}, {{19.0680, 19.1208}, {20.6880, 21.0968}})};

    std::vector<Tensor> com_tensors = {*a.grad_, *b.grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, Linear_1) {
    Tensor x = Tensor({1, 3}, {1, 2, 3}, {}, true);

    Linear* l = new Linear(3, 2);
    l->weight_ = new Weight(Tensor({2, 3}, 1.0f));
    l->bias_->fill(.0f);

    Linear* ll = new Linear(2, 1);
    ll->weight_ = new Weight(Tensor({1, 2}, 1.0f));
    ll->bias_->fill(.0f);

    Sequential seq = Sequential({l, ll});
    Tensor z = seq(x);

    z.grad_->data_ = {1};
    z.backward();

    std::vector<Tensor> exp_tensors = {
        Tensor({2, 3}, {1, 2, 3, 1, 2, 3}),
        Tensor({1, 2}, {1, 1}),
        Tensor({1, 2}, {6, 6}),
        Tensor({1, 1}, {1.0f}),
    };

    std::vector<Tensor> com_tensors = {l->weight_->grad(), *l->bias_->grad_, ll->weight_->grad(), *ll->bias_->grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, LinearRelu_1) {
    Tensor x = Tensor({1, 3}, {-1, -2, 8.3}, {}, true);

    Linear* l = new Linear(3, 2);
    l->weight_ = new Weight(Tensor({2, 3}, {-0.5, 0.5, 0.6, 0.2, -0.3, -0.34}));
    l->bias_->fill(.0f);

    Linear* ll = new Linear(2, 1);
    ll->weight_ = new Weight(Tensor({1, 2}, 1.0f));
    ll->bias_->fill(.0f);
    Sequential seq = Sequential({l, new ReLU(), ll});
    Tensor out = seq(x);

    out.grad_->fill(1);
    out.backward();

    std::vector<Tensor> exp_tensors = {
        Tensor({2, 3}, {-1.0000, -2.0000, 8.3000, 0, 0, 0}),
        Tensor({1, 2}, {1, 0}),
        Tensor({1, 2}, {4.48, 0}),
        Tensor({1, 1}, {1.0f}),
    };

    Tensor q = l->weight_->grad();

    std::vector<Tensor> com_tensors = {l->weight_->grad(), *l->bias_->grad_, ll->weight_->grad(), *ll->bias_->grad_};

    for (int i = 0; i < exp_tensors.size(); i++) {
        ASSERT_TRUE(compare_tensors(exp_tensors[i], com_tensors[i], i));
    }
}

TEST(GRAD, Sigmoid_1) {
    Tensor* x = new Tensor({1}, 1.0f, {}, true);
    Sigmoid s = Sigmoid();
    Tensor* q = s.forward(x);

    Tensor exp = Tensor({1}, 0.1966f);

    q->grad_->fill(1);
    q->backward();

    ASSERT_TRUE(compare_tensors(*x->grad_, exp, 0));
}

TEST(GRAD, LeakyRelu_1) {
    Tensor* x = new Tensor({1}, 0.534f, {}, true);

    LeakyReLU lr1 = LeakyReLU(.01);

    Tensor* q = lr1.forward(x);

    Tensor exp = Tensor({1}, 1.0f);

    q->grad_->fill(1);
    q->backward();

    ASSERT_TRUE(compare_tensors(*x->grad_, exp, 0));
}

/*
 * tests with 3+dim tensors
 */

TEST(HIGHDIM_TENSOR_OPERATIONS, HighDimMatmul) {
    Vec2D data = load_test_data();

    vector<Tensor> examples = {
        Tensor({2, 3, 4, 5}, data[0]),
        Tensor({2, 3, 5, 2}, data[1]),
    };

    vector<Tensor> results = {Tensor({2, 3, 4, 2}, data[2])};

    for (int i = 0; i < examples.size(); i += 2) {
        Tensor a = examples[i];
        Tensor b = examples[i + 1];

        Tensor computed = a.matmul(b);

        Tensor expected = results[i / 2];
        
        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(HIGHDIM_TENSOR_OPERATIONS, HighDimTranspose) {
    Vec2D data = load_test_data();

    vector<Tensor> examples = {
        Tensor({2, 3, 2, 4}, data[0]),
    };

    vector<Tensor> results = {Tensor({2, 3, 4, 2}, data[1])};

    for (int i = 0; i < examples.size(); i++) {
        Tensor a = examples[i];

        Tensor computed = a.transpose();

        Tensor expected = results[i];

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(HIGHDIM_TENSOR_OPERATIONS, HighDimCorrelate) {
    Vec2D data = load_test_data();

    vector<Tensor> examples = {
        Tensor({1, 2, 4, 4}, data[0], {}, true),
        Tensor({1, 2, 2, 2}, data[1], {}, true),
    };

    vector<Tensor> results = {Tensor({1, 1, 3, 3}, data[2])};

    for (int i = 0; i < examples.size(); i += 2) {
        Tensor a = examples[i];
        Tensor b = examples[i + 1];

        Tensor computed = a.correlate(b);

        Tensor expected = results[i / 2];

        ASSERT_TRUE(computed == expected) << std::to_string(i) << " tensor; expected:\n"
                                          << expected.to_string() << "; got:\n" + computed.to_string();
    }
}

TEST(NN, Conv2D_1) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 1, 2, 2}, {1, 2, 4, 5}, {}, true);
    Conv2D m = Conv2D(1, 3, {1, 1}, {1, 1}, {0, 0});
    Tensor weight_tensor = Tensor(m.weight_->weight_->shape_, data[0], {}, true);
    m.weight_ = new Weight(weight_tensor);
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);
    computed->grad_->fill(data[1]);
    computed->backward();

    Tensor expected = Tensor({1, 3, 2, 2}, data[2]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({3, 1, 1, 1}, data[3]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 1, 2, 2}, data[4]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_2) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 2, 3, 3}, data[0], {}, true);
    Conv2D m = Conv2D(2, 3, {2, 2}, {1, 1}, {0, 0});
    Tensor mw = Tensor(m.weight_->weight_->shape_, 1, {}, true);
    mw.data_[0] = 2;
    mw.data_[1] = 3;
    m.weight_ = new Weight(mw);
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill(data[1]);
    computed->backward();

    Tensor expected = Tensor({1, 3, 2, 2}, data[2]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({3, 2, 2, 2}, data[3]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 2, 3, 3}, data[4]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_3) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 2, 2, 2}, data[0], {}, true);
    Conv2D m = Conv2D(2, 3, {2, 2}, {1, 1}, {0, 0});
    Tensor mw = Tensor(m.weight_->weight_->shape_, data[1], {}, true);
    m.weight_ = new Weight(mw);
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);

    Tensor expected = Tensor({1, 3, 1, 1}, data[2]);

    computed->grad_->fill(data[3]);
    computed->backward();

    ASSERT_TRUE(compare_tensors(expected, *computed));
}

TEST(NN, Conv2D_4) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 1, 5, 5}, 3.0, {}, true);
    Conv2D m = Conv2D(1, 1, {3, 3}, {2, 2}, {0, 0});
    Tensor mw = Tensor(m.weight_->weight_->shape_, 1, {}, true);
    mw.data_[0] = 2;
    mw.data_[1] = 3;
    m.weight_ = new Weight(mw);
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill({7, 1, 2, 5});
    computed->backward();

    Tensor expected = Tensor({1, 1, 2, 2}, data[0]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({1, 1, 3, 3}, data[1]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 1, 5, 5}, data[2]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_5) {
    Vec2D data = load_test_data();

    Vec1D xval = data[0];

    Tensor* x = new Tensor({1, 1, 8, 8}, xval, {}, true);

    Conv2D m = Conv2D(1, 1, {3, 3}, {3, 3}, {0, 0, 0});
    m.weight_ = new Weight(Tensor(m.weight_->weight_->shape_, data[1], {}, true));
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill(data[2]);
    computed->backward();

    Tensor expected = Tensor({1, 1, 2, 2}, data[3]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({1, 1, 3, 3}, data[4]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 1, 8, 8}, data[5]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_6) {
    Vec2D data = load_test_data();

    Vec1D xval = data[0];

    Tensor* x = new Tensor({1, 2, 8, 8}, xval, {}, true);

    Conv2D m = Conv2D(2, 6, {3, 3}, {3, 3}, {0, 0, 0});
    m.weight_ = new Weight(Tensor(m.weight_->weight_->shape_, data[1], {}, true));
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill(data[2]);
    computed->backward();

    Tensor expected = Tensor({1, 6, 2, 2}, data[3]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({6, 2, 3, 3}, data[4]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 2, 8, 8}, data[5]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_7) {
    Vec2D data = load_test_data();

    Vec1D xval = data[0];

    Tensor* x = new Tensor({1, 2, 8, 8}, xval, {}, true);

    Conv2D m = Conv2D(2, 2, {3, 3}, {3, 3}, {0, 0, 0});
    m.weight_ = new Weight(Tensor(m.weight_->weight_->shape_, data[1], {}, true));
    m.bias_ = new Tensor(m.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill(data[2]);
    computed->backward();

    Tensor expected = Tensor({1, 2, 2, 2}, data[3]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({2, 2, 3, 3}, data[4], {}, false);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 2, 8, 8}, data[5]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_8) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 1, 2, 2}, {1, 2, 4, 5}, {}, true);
    Conv2D m = Conv2D(1, 3, {1, 1}, {1, 1}, {0, 0});
    Tensor weight_tensor = Tensor(m.weight_->weight_->shape_, data[0], {}, true);
    m.weight_ = new Weight(weight_tensor);
    m.bias_ = new Tensor({1, 3, 1, 1}, {1, 2, 3}, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill(data[1]);
    computed->backward();

    Tensor expected = Tensor({1, 3, 2, 2}, data[2]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({3, 1, 1, 1}, data[3]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_f_grad));

    Tensor exp_b_grad = Tensor({1, 3, 1, 1}, data[4]);
    ASSERT_TRUE(compare_tensors(*m.bias_->grad_, exp_b_grad));

    Tensor exp_x_grad = Tensor({1, 1, 2, 2}, data[5]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Conv2D_9) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({3, 4, 4, 4}, data[0], {}, true);
    Conv2D m = Conv2D(4, 2, {2, 2}, {1, 1}, {0, 0});
    Tensor weight_tensor = Tensor(m.weight_->weight_->shape_, data[1], {}, true);
    m.weight_ = new Weight(weight_tensor);
    m.bias_ = new Tensor({1, 2, 1, 1}, 0, {}, true);

    Tensor* computed = m.forward(x);

    computed->grad_->fill(computed->data_);
    computed->backward();

    Tensor expected = Tensor({3, 2, 3, 3}, data[2]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_w_grad = Tensor(m.weight_->weight_->shape_, data[3]);
    ASSERT_TRUE(compare_tensors(m.weight_->grad(false), exp_w_grad));

    Tensor exp_x_grad = Tensor(x->shape_, data[4]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Deep_Conv2D_1) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 2, 3, 3}, data[0], {}, true);

    Conv2D f_layer = Conv2D(2, 3, {2, 2}, {1, 1}, {0, 0});
    Tensor f_weight = Tensor(f_layer.weight_->weight_->shape_, data[1], {}, true);
    f_layer.weight_ = new Weight(f_weight);
    f_layer.bias_ = new Tensor(f_layer.bias_->shape_, 0.0f, {}, true);

    Conv2D s_layer = Conv2D(3, 2, {2, 2}, {1, 1}, {0, 0});
    Tensor s_weight = Tensor(s_layer.weight_->weight_->shape_, data[2], {}, true);
    s_layer.weight_ = new Weight(s_weight);
    s_layer.bias_ = new Tensor(s_layer.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = f_layer.forward(x);
    computed = s_layer.forward(computed);

    computed->grad_->fill(data[3]);
    computed->backward();

    Tensor expected = Tensor({1, 2, 1, 1}, data[4]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({3, 2, 2, 2}, data[5]);
    ASSERT_TRUE(compare_tensors(f_layer.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 2, 3, 3}, data[6]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Deep_Conv2D_2) {
    Vec2D data = load_test_data();

    Vec1D x_val = data[0];
    Tensor* x = new Tensor({1, 2, 8, 8}, x_val, {}, true);

    Conv2D l = Conv2D(2, 3, {2, 2}, {2, 2}, {0, 0});
    l.weight_ = new Weight(Tensor(l.weight_->weight_->shape_, data[1], {}, true));
    l.bias_ = new Tensor(l.bias_->shape_, 0.0f, {}, true);

    Conv2D ll = Conv2D(3, 6, {3, 3}, {2, 2}, {0, 0});
    ll.weight_ = new Weight(Tensor(ll.weight_->weight_->shape_, data[2], {}, true));
    ll.bias_ = new Tensor(ll.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = ll.forward(l.forward(x));

    computed->grad_->fill(data[3]);
    computed->backward();

    Tensor expected = Tensor({1, 6, 1, 1}, data[4]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_ll_grad = Tensor({6, 3, 3, 3}, data[5]);
    ASSERT_TRUE(compare_tensors(ll.weight_->grad(false), exp_ll_grad));

    Tensor exp_x_grad = Tensor({1, 2, 8, 8}, data[6]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

TEST(NN, Deep_Conv2D_3) {
    Vec2D data = load_test_data();

    Tensor* x = new Tensor({1, 2, 8, 8}, data[0], {}, true);

    Conv2D f_layer = Conv2D(2, 3, {3, 3}, {3, 3}, {1, 1});
    Tensor f_weight = Tensor(f_layer.weight_->weight_->shape_, data[1], {}, true);
    f_layer.weight_ = new Weight(f_weight);
    f_layer.bias_ = new Tensor(f_layer.bias_->shape_, 0.0f, {}, true);

    Conv2D s_layer = Conv2D(3, 7, {2, 2}, {3, 3}, {2, 2});
    Tensor s_weight = Tensor(s_layer.weight_->weight_->shape_, data[2], {}, true);
    s_layer.weight_ = new Weight(s_weight);
    s_layer.bias_ = new Tensor(s_layer.bias_->shape_, 0.0f, {}, true);

    Tensor* computed = f_layer.forward(x);
    computed = s_layer.forward(computed);

    computed->grad_->fill(data[3]);
    computed->backward();

    Tensor expected = Tensor({1, 7, 2, 2}, data[4]);
    ASSERT_TRUE(compare_tensors(expected, *computed));

    Tensor exp_f_grad = Tensor({3, 2, 3, 3}, data[5]);
    ASSERT_TRUE(compare_tensors(f_layer.weight_->grad(false), exp_f_grad));

    Tensor exp_x_grad = Tensor({1, 2, 8, 8}, data[6]);
    ASSERT_TRUE(compare_tensors(*x->grad_, exp_x_grad));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
