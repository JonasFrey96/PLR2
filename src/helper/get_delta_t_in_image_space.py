import torch


def get_delta_t_in_image_space(t_tar, t_src, f_x, f_y):
    """
    DeepIM equation 3 
    Inputs: target translation bs * [x,y,z]
            source translation bs * [x,y,z]
            f_x                bs * f_x 
            f_x                bs * f_y
    Outputs: v_x v_y v_z
    """
    v_x = (f_x[:, 0] * (t_tar[:, 0] / t_tar[:, 2] - t_src[:, 0] / t_src[:, 2]))
    v_y = (f_y[:, 0] * (t_tar[:, 1] / t_tar[:, 2] - t_src[:, 1] / t_src[:, 2]))
    v_z = (torch.log(t_src[:, 2] / t_tar[:, 2]))

    return torch.stack([v_x, v_y, v_z], dim=1)


if __name__ == "__main__":
    # batch test
    t_tar = torch.ones((100, 3))
    t_src = torch.ones((100, 3))
    f_x = torch.ones((100, 1))
    f_y = torch.ones((100, 1))
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)

    f_x = torch.tensor([[1]], dtype=torch.float32)
    f_y = torch.tensor([[1]], dtype=torch.float32)

    # sense checking
    t_tar = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    t_src = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print not moveing', v)
    t_src = torch.tensor([[0.1, 0, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move up', v)
    t_src = torch.tensor([[-0.1, 0, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move down', v)
    t_src = torch.tensor([[0, -0.1, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move left', v)
    t_src = torch.tensor([[0, 0.1, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move right', v)

    t_src = torch.tensor([[0.1, 0, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move up+ back', v)
    t_src = torch.tensor([[-0.1, 0, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move down+ back', v)
    t_src = torch.tensor([[0, -0.1, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move left+ back', v)
    t_src = torch.tensor([[0, 0.1, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, f_x, f_y)
    print('print move right+ back', v)

    # looks good to me. Maybe add some testing with real camera and images from object in different poses
