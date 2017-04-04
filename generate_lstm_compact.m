%
% generate_lstm_compact.m
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/09/2016
%
% generate a sequence given a LSTM network 
% note: for the optimized version (single W, U matrices)
%

function text = generate_lstm_compact(U, W, b, Why, by, l, h, c)

    text = [];
    codes = eye(size(Why, 1));

    K = size(h, 1);
    for i=1:l-1
        
        y = Why * h + by;
        probs = exp(y)./sum(exp(y));
        cdf = cumsum(probs);

        r = rand();
        sample = min(find(r <= cdf));
        text = [text char(sample)];

        %update hidden state
        x = codes(sample, :)';

        iofc = W * x + U * h + b;

        i = sigmoid(iofc(1:K, :));
        f =  sigmoid(iofc(2*K+1:3*K, :));
        cc =  tanh(iofc((3*K+1):end, :));
        o =  sigmoid(iofc(K+1:2*K, :));
        c = tanh(i .* cc + f .* c);

        h = o .* c;
        
    end

end