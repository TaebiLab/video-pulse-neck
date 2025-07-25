function [A, S] = jade(X, m, Wprev)
    [n, T] = size(X);
    nem = m;
    seuil = 1 / sqrt(T) / 100;
    if m < n
        [D, U] = eig(X * X' / T);
        Diag = diag(D);
        [~, k] = sort(Diag);
        pu = Diag(k);
        ibl = sqrt(pu(n - m + 1:n) - mean(pu(1:n - m)));
        bl = ones(m, 1) ./ ibl;
        W = diag(bl) * U(:, k(n - m + 1:n))';
        IW = U(:, k(n - m + 1:n)) * diag(ibl);
    else
        IW = sqrtm(X * X' / T);
        W = inv(IW);
    end

    Y = W * X;
    R = (Y * Y') / T;
    C = (Y * Y.') / T;
    Q = zeros(m * m * m * m, 1);
    index = 0;

    for lx = 1:m
        Y1 = Y(lx, :);
        for kx = 1:m
            Yk1 = Y1 .* conj(Y(kx, :));
            for jx = 1:m
                Yjk1 = Yk1 .* conj(Y(jx, :));
                for ix = 1:m
                    Q(index + 1) = (Yjk1 / sqrt(T)) * (Y(ix, :)' / sqrt(T)) - R(ix, jx) * R(lx, kx) - R(ix, kx) * R(lx, jx) - C(ix, lx) * conj(C(jx, kx));
                    index = index + 1;
                end
            end
        end
    end

    [D, U] = eig(reshape(Q, m * m, m * m));
    Diag = abs(diag(D));
    [~, K] = sort(Diag);
    la = Diag(K);
    M = zeros(m, nem * m);
    h = m * m;
    for u = 1:m:nem * m
        Z = reshape(U(:, K(h)), m, m);
        M(:, u:u + m - 1) = la(h) * Z;
        h = h - 1;
    end

    B = [1, 0, 0; 0, 1, 1; 0, -1i, 1i];
    Bt = B';

    encore = true;
    if Wprev == 0
        V = eye(m);
    else
        V = inv(Wprev);
    end

    while encore
        encore = false;
        for p = 1:m - 1
            for q = p + 1:m
                Ip = p:m:nem * m;
                Iq = q:m:nem * m;
                g = [M(p, Ip) - M(q, Iq); M(p, Iq); M(q, Ip)];
                temp1 = g * g';
                temp2 = B * temp1;
                temp = temp2 * Bt;
                [D, vcp] = eig(real(temp));
                [~, K] = sort(diag(D));
                angles = vcp(:, K(3));
                if angles(1) < 0
                    angles = -angles;
                end
                c = sqrt(0.5 + angles(1) / 2);
                s = 0.5 * (angles(2) - 1i * angles(3)) / c;

                if abs(s) > seuil
                    encore = true;
                    pair = [p, q];
                    G = [c, -conj(s); s, c];
                    V(:, pair) = V(:, pair) * G;
                    M(pair, :) = G' * M(pair, :);
                    temp1 = c * M(:, Ip) + s * M(:, Iq);
                    temp2 = -conj(s) * M(:, Ip) + c * M(:, Iq);
                    M(:, Ip) = temp1;
                    M(:, Iq) = temp2;
                end
            end
        end
    end

    A = IW * V;
    S = V' * Y;
end