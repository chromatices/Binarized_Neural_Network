def shift_adamax(model: model, lr=2 ** -10):
    # 수정해야할 부분이 많은 코드!
    '''todo
    1. adamax 최대한 많이 숙지하기. 기존 adamax에서 파생된 개념인 만큼 많은것을 가져와야함
    2. 모델을 뜯던 파라미터를 뜯던 가중치를 뜯어야함
    3. 변수가 음수값을 가지면 unit.64 , 양수값을 가지면 int.64 형태로 retyping 하기
    4. adamax가 forward한 최종적인 값으로 한건지, 1batch의 최종적인 결과값으로 한건지 찾아보기
    5. 최종적으로 수렴하는 가중치(m_hat)를 만들기 위해 직전의 가중치와 값이 똑같으면 m_hat
    '''

    mod = sys.modules[__name__]                 # 자동 변수화하기 위한 코드. weight 층이 몇개나 있을지 모르기 때문에 자동으로 변수화 하거나
                                                # 리스트처리를 해야함. 우선 뼈대를 완성하기 위해 변수에 저장하기 위해 작업을 시행함.
    i=1
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            if name.endswith("weight"):
                i+=1
                setattr(mod,'weight{}'.format(i),param)
                weightt = param
                info = name

    b1, b2 = 1 - 2 ** -3, 1 - 2 ** -10
    grad = model.grad                       # 모델에서 gradient 추출해야함.
    weight = b1 * weight + (1 - b1) * grad  # first momentum
    v = max(b2 * v, abs(grad))              # second momentum

    var1 = lr >> (1 - b1)                   # 나중에 수정해야함 float 단위에서는 bit shift 불가능.
    var2 = weight >> 1 / v

    parameters = parameters - (var1 * var2)

    state = optimizer.state_dict['state']