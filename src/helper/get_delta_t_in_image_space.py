import torch


def get_delta_t_in_image_space(t_tar, t_src, fx, fy):
    """
    only works for positive z values !

    DeepIM equation 3
    Inputs: target translation bs * [x,y,z]
            source translation bs * [x,y,z]
            fx                bs * fx 
            fx                bs * fy
    Outputs: v_x v_y v_z
    """
    v_x = (fx[:, 0] * (torch.true_divide(t_tar[:, 0], t_tar[:, 2]
                                         ) - torch.true_divide(t_src[:, 0], t_src[:, 2])))
    v_y = (fy[:, 0] * (torch.true_divide(t_tar[:, 1], t_tar[:, 2]
                                         ) - torch.true_divide(t_src[:, 1], t_src[:, 2])))

    print('log in', )
    v_z = torch.true_divide(t_src[:, 2], t_tar[:, 2])
    if torch.min(v_z) < 0:
        print('Error negative number in log take abs but check input')
        v_z = torch.abs(v_z)
    v_z = torch.log(v_z)
    return torch.stack([v_x, v_y, v_z], dim=1)


def get_delta_t_in_euclidean(v, t_src, fx, fy, device):
    """ convert inital object pose and predicted image coordinates to euclidean translation
    only works for positive z values !

    Args:
        v (torch.Tensor):     bs x [x,y,z]
        t_src (torch.Tensor): inital object position bs * [x,y,z]
        fx (torch.Tensor):   bs * fx 
        fy (torch.Tensor):   bs * fy

    Returns:
        torch.Tensor: target object position bs * [x,y,z]
    """
    # alternative implementation override t_src for intrinisc runtime capable or pass input tensor into function
    t_pred_tar = torch.zeros(t_src.shape, device=device)
    t_pred_tar[:, 2] = torch.true_divide(
        t_src.clone()[:, 2], torch.exp(v[:, 2]))
    t_pred_tar[:, 0] = (torch.true_divide(v.clone()[:, 0], fx.clone()[:, 0]) +
                        torch.true_divide(t_src.clone()[:, 0], t_src.clone()[:, 2])) * t_pred_tar.clone()[:, 2]
    t_pred_tar[:, 1] = (torch.true_divide(v.clone()[:, 1], fy.clone()[:, 0]) +
                        torch.true_divide(t_src.clone()[:, 1], t_src.clone()[:, 2])) * t_pred_tar.clone()[:, 2]
    return t_pred_tar


if __name__ == "__main__":
    # batch test
    t_tar = torch.ones((100, 3))
    t_src = torch.ones((100, 3))
    fx = torch.ones((100, 1))
    fy = torch.ones((100, 1))
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)

    fx = torch.tensor([[1122.4123]], dtype=torch.float32)
    fy = torch.tensor([[8813.123123]], dtype=torch.float32)

    print('Test from global to image')
    # sense checking
    t_tar = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    t_src = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print not moveing', v)
    t_src = torch.tensor([[0.1, 0, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move up', v)
    t_src = torch.tensor([[-0.1, 0, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move down', v)
    t_src = torch.tensor([[0, -0.1, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move left', v)
    t_src = torch.tensor([[0, 0.1, 1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move right', v)

    t_src = torch.tensor([[0.1, 0, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move up+ back', v)
    t_src = torch.tensor([[-0.1, 0, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move down+ back', v)
    t_src = torch.tensor([[0, -0.1, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move left+ back', v)
    t_src = torch.tensor([[0, 0.1, 1.1]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)
    print('print move right+ back', v)

    # looks good to me. Maybe add some testing with real camera and images from object in different poses
    print('Test image to global')
    t_src = torch.tensor([[0.56, 0.12, 1.12]], dtype=torch.float32)
    t_tar = torch.tensor([[0.99, .312, 0.127]], dtype=torch.float32)
    v = get_delta_t_in_image_space(t_tar, t_src, fx, fy)

    t_pred_tar = get_delta_t_in_euclidean(v, t_src, fx, fy)
    print(f'input tar {t_tar}, output {t_pred_tar}')

    bs = 100

    t_src = torch.normal(mean=torch.zeros((bs, 3)), std=torch.ones((bs, 3)))
    t_tar = torch.normal(mean=torch.zeros((bs, 3)), std=torch.ones((bs, 3)))
    t_src[:, 2] = torch.abs(t_src[:, 2])
    t_tar[:, 2] = torch.abs(t_tar[:, 2])
    v = get_delta_t_in_image_space(
        t_tar, t_src, fx.repeat((bs, 1)), fy.repeat((bs, 1)))

    t_pred_tar = get_delta_t_in_euclidean(
        v, t_src, fx.repeat((bs, 1)), fy.repeat((bs, 1)))

    print('average error converting back and forward:',
          torch.sum(t_tar - t_pred_tar, dim=0) / bs)
