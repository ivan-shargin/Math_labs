def solve_diags_3(A, f):
    """Функция реализует метод прогонки СЛАУ.
    Принимает матрицу А левой части и вектор f правой части,
    необходимую точность. Возвращает вектор решения"""
    import numpy as np
    N = A.shape[0]
    d = f
    c = np.diagonal(A, offset=1)
    b = np.diagonal(A)
    a = np.diagonal(A, offset=-1)

    L = np.tril(A, k=-2)
    U = np.triu(A, k=2)
    if np.linalg.norm(L) > 0 or np.linalg.norm(U) > 0:
        print("Not 3x diagonal matrix!")
        return(-1)

    answer = [0] * N
    y = []
    y.append(b[0])
    alpha = []
    alpha.append(-c[0] / y[0])
    beta = []
    beta.append(d[0]/y[0])

    for i in range(1, N-1):
        y.append(b[i] + a[i-1] * alpha[-1])
        alpha.append(-c[i] / y[-1])
        beta.append((d[i] - a[i-1] * beta[-1]) / y[-1])
    y.append(b[N-1] + a[N-2] * alpha[-1])
    beta.append((d[N-1] - a[N-2] * beta[-1]) / y[-1])

    answer[-1] = beta[-1]
    for i in range(N-2, -1, -1):
        answer[i] = alpha[i] * answer[i+1] + beta[i]
    answer = np.array(answer)

    return(answer)
