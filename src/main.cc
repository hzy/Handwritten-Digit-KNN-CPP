#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include "incbin.h"

INCBIN(_train_images, "mnist/train-images-idx3-ubyte");
INCBIN(_test_images, "mnist/t10k-images-idx3-ubyte");
INCBIN(_train_labels, "mnist/train-labels-idx1-ubyte");
INCBIN(_test_labels, "mnist/t10k-labels-idx1-ubyte");

constexpr const unsigned char *train_images_bin_start = g_train_images_data + 16;
constexpr const unsigned char *test_images_bin_start = g_test_images_data + 16;
constexpr const unsigned char *train_labels_bin_start = g_train_labels_data + 8;
constexpr const unsigned char *test_labels_bin_start = g_test_labels_data + 8;

constexpr int number_of_rows = 28;
constexpr int number_of_columns = 28;
constexpr int image_size = number_of_rows * number_of_columns * sizeof(u_int8_t);

#ifdef DEBUG
int number_of_train_images = 0;
int number_of_test_images = 0;
#else
int number_of_train_images = (g_train_images_end - train_images_bin_start) / image_size;
int number_of_test_images = (g_test_images_end - test_images_bin_start) / image_size;
#endif

int wrong_count = 0;
int all_count = 0;
std::mutex wrong_count_mutex;
std::mutex all_count_mutex;

struct distance : std::pair<u_int8_t, double>
{
    using std::pair<u_int8_t, double>::pair;
    bool operator<(const distance &str) const
    {
        return this->second < str.second;
    }
    friend std::ostream &operator<<(std::ostream &out, const distance &c)
    {
        out << "label: " << (char)(c.first + '0') << ", "
            << "distance: " << c.second;
        return out;
    }
};

double minkowski_distance(std::vector<uint8_t> a, std::vector<uint8_t> b, int p)
{
    double sum = 0;
    for (int i = 0, l = std::min(a.size(), b.size()); i < l; i++)
        sum += std::pow(std::abs(a[i] - b[i]), p);

    return std::pow(sum, 1 / (p + 0.0));
}

u_int8_t predict(std::vector<u_int8_t> test_image, int k, int p)
{
    std::vector<distance> distances(number_of_train_images);
    const u_int8_t *base = train_images_bin_start;

    for (int i = 0; i < number_of_train_images; i++)
    {
        std::vector<u_int8_t> train_image(base, base + image_size);
        double d = minkowski_distance(train_image, test_image, p);
        distances[i] = {train_labels_bin_start[i], d};

        base += image_size;
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> top_k(k);
    for (int i = 0; i < k; i++)
        top_k[distances[i].first]++;

    u_int8_t result = 0;
    for (int i = 1; i < k; i++)
        if (top_k[i] > top_k[result])
            result = i;

    return result;
}

void worker(const u_int8_t *base, int start, int count, int k, int p)
{
    printf("A worker is started!\n");
    int local_wrong_count = 0;
    int local_all_count = 0;

    base += image_size * start;
    for (int i = 0; i < count; i++)
    {
        std::vector<u_int8_t> test_image(base, base + image_size);
        u_int8_t result = predict(test_image, k, p);
        u_int8_t right = test_labels_bin_start[start + i];

        local_all_count++;

        if (result != right)
            local_wrong_count++;

        base += image_size;
    }

    {
        std::lock_guard<std::mutex> guard_all(all_count_mutex);
        std::lock_guard<std::mutex> guard_wrong(wrong_count_mutex);
        all_count += local_all_count;
        wrong_count += local_wrong_count;
    }
}

int main(int argc, char **argv)
{
#ifdef DEBUG
    if (argc != 5)
    {
        std::cout << "Debug Usage: " << argv[0] << " K P TrainImageNum TestImageNum" << std::endl;
        return 1;
    }
    number_of_train_images = atoi(argv[3]);
    number_of_test_images = atoi(argv[4]);
#else
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " K P" << std::endl;
        return 1;
    }
#endif

    std::cout << "number_of_train_images: " << number_of_train_images << std::endl;
    std::cout << "number_of_test_images: " << number_of_test_images << std::endl;

    unsigned int cpu_number = std::thread::hardware_concurrency();
    unsigned int load_per_cpu = number_of_test_images / cpu_number;

    std::vector<std::thread *> threads(cpu_number);
    for (unsigned int i = 0; i < cpu_number; i++)
        threads[i] = new std::thread(worker, test_images_bin_start, i * load_per_cpu, (int)load_per_cpu, atoi(argv[1]), atoi(argv[2]));

    for (unsigned int i = 0; i < cpu_number; i++)
        threads[i]->join();

    std::cout << wrong_count << " / " << all_count << std::endl;
    std::cout << "acc: " << 100 - wrong_count / (all_count + 0.0) * 100 << '%' << std::endl;
}
