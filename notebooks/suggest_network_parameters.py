def find_convolution_configs(input_dim, kernel_sizes, num_layers=5):

    configurations = []

    def calc_stride(input_size, kernel_size):
        valid_strides = []
        for stride in range(2, 8):  # Iterate over possible strides
            if (input_size - kernel_size) % stride == 0:
                output_size = (input_size - kernel_size) / stride + 1
                if output_size > 0:
                    valid_strides.append((stride, output_size))
        return valid_strides

    def dfs(layer, current_size, current_kernels, current_strides):
        if layer == num_layers:
            size_after_conv = current_size
            config = {
                "kernel_sizes": current_kernels[:],
                "strides": current_strides[:],
                "size_after_conv": size_after_conv
            }
            configurations.append(config)
            return

        for k in kernel_sizes:
            for stride, next_size in calc_stride(current_size, k):
                current_kernels.append(k)
                current_strides.append(stride)

                dfs(layer + 1, next_size, current_kernels, current_strides)

                current_kernels.pop()
                current_strides.pop()

    dfs(0, input_dim, [], [])

    for i, conf in enumerate(configurations):
        print(f"Config {i+1}:\n{conf}\n--------------------------------------\n")



def validate_configuration(input_dim, kernel_sizes, strides):
    current_size = input_dim

    if len(kernel_sizes) != len(strides):
        print('NOT VALID')
        return False

    for k, s in zip(kernel_sizes, strides):
        if s <= 0 or k <= 0:
            print('NOT VALID')
            return

        if (current_size - k) % s != 0:
            print('NOT VALID')
            return

        next_size = (current_size - k) / s + 1
        if next_size <= 0:
            print('NOT VALID')
            return

        current_size = next_size

    print('VALID')


if __name__ == '__main__':
    input_dim = 44000
    possible_kernel_sizes = [3, 5, 11, 15]
    num_layers = 5

    find_convolution_configs(
        input_dim, possible_kernel_sizes, num_layers
    )

    input_dim = 44000
    test_kernel_sizes = [15, 11, 5, 3, 5]
    test_strides = [5, 3, 3, 7, 3]

    validate_configuration(input_dim, test_kernel_sizes, test_strides)
