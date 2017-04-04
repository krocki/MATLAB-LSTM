%
% lstm.m
%
% LSTM code - compacted
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/09/2016
%
%

%read raw byte stream
data = read_raw('alice29.txt');

text_length = size(data, 1);

% alphabet
symbols = unique(data);
alphabet_size = size(symbols, 1);
ASCII_SIZE = 256;

% n_in - size of the alphabet, ex. 4 (ACGT)
n_in = ASCII_SIZE;
% n_out = n_in - predictions have the same size as inputs
n_out = n_in;

codes = eye(n_in);

max_iterations = text_length;
max_epochs = 1000;

observations = zeros(1, text_length);
perc = round(text_length / 100);
show_every = 5; %show stats every show_every s

% hyperparameters
hidden_size = 16; % size of hidden layer of neurons
seq_length = 5; % number of steps to unroll the RNN for
learning_rate = 1e-1;
vocab_size = n_in;

% model parameters
W = randn(hidden_size * 4, vocab_size) * 0.01; % x to gates
U = randn(hidden_size * 4, hidden_size) * 0.01; % h_prev to gates
b = zeros(hidden_size * 4, 1); %gates' biases

Why = randn(vocab_size, hidden_size) * 0.01; % hidden to output
by = zeros(vocab_size, 1); % output bias

h = zeros(hidden_size, seq_length);
c = zeros(hidden_size, seq_length); % memory cells
tc = zeros(hidden_size, seq_length); % temp tanh cell
iofc = zeros(hidden_size * 4, seq_length); % input gates

%adagrad memory
mWhy = zeros(size(Why));
mby = zeros(size(by));

mW = zeros(size(W));
mU = zeros(size(U));
mb = zeros(size(b));

target = zeros(vocab_size, seq_length);
y = zeros(vocab_size, seq_length);
dy = zeros(vocab_size, seq_length);
probs = zeros(vocab_size, seq_length);

%using log2 (bits), initial guess
smooth_loss = - log2(1.0 / alphabet_size);
loss_history = [];

h_act_history = [];
c_act_history = [];

%reset timer
tic

h = rand(size(h));
K = hidden_size; % hidden count - to make it shorter

for e = 1:max_epochs
    
    %set some random context
    h(:, 1) = tanh(zeros(size(h(:, 1))));
    c(:, 1) = tanh(zeros(size(c(:, 1))));
    %or zeros
    %h(:, 1) = zeros(size(h(:, 1)));
    
    beginning = randi([2 1+seq_length]); %randomize starting point

    for ii = beginning:seq_length:max_iterations - seq_length
        
        % reset grads
        dby = zeros(size(by));
        dWhy = zeros(size(Why));
        dy = zeros(size(target));
        diofc = zeros(size(iofc, 1), 1);
        
        dU = zeros(size(U));
        dW = zeros(size(W));     
        db = zeros(size(b));
        
        dcnext = zeros(size(c(:, 1)));
        dhnext = zeros(size(h(:, 1)));
        
        % get next symbol
        xs(:, 1:seq_length) = codes(data(ii - 1:ii + seq_length - 2), :)';
        target(:, 1:seq_length) = codes(data(ii:ii + seq_length - 1), :)';
        
        observations = char(data(ii - 1:ii + seq_length - 2))';
        t_observations = char(data(ii:ii + seq_length - 1))';
        
        % forward pass:
        
        loss = 0;
        
        for t = 2:seq_length
            
            %LSTM gates
            %note: this code is optimized a bit, so all Ws and Us reside in big matrices,
            %it's better to read the non-optimized version for understanding
            %W = [Wi Wo Wf Wc], iofc - input, output, forget, candidate outputs
            iofc(:, t) = W * xs(:, t) + U * h(:, t - 1) + b; 
            iofc(1:K*3, t) = sigmoid(iofc(1:K*3, t)); %iof - sigmoid
            iofc(K*3+1:end, t) = tanh(iofc(K*3+1:end, t)); %c - tanh

            %new context state, c(t) = tanh(i(t) * cc(t) + f(t) * c(t-1))
            c(:, t) = tanh( iofc(1:K, t) .* iofc(K*3+1:end, t) + ...
                            iofc(K*2+1:K*3, t) .* c(:, t - 1));
            
            %new hidden state
            h(:, t) = iofc(K*1+1:K*2, t) .* c(:, t);
            
            % update y
            y(:, t) = Why * h(:, t) + by;
            
            % compute probs
            probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
            
            % cross-entropy loss, sum logs of probabilities of target outputs
            loss = loss + sum( -log2(probs(:, t)) .* target(:, t));
            
        end
        
        %bits/symbol
        loss = loss/seq_length;
        
        % backward pass:
        for t = seq_length: - 1:2
            
            % dy (global error)
            dy(:, t) = probs(:, t) - target(:, t); % %dy[targets[t]] -= 1 # backprop into y
            dWhy = dWhy + dy(:, t) * h(:, t)'; %dWhy += np.doutt(dy, hs[t].T)
            dby = dby + dy(:, t); % dby += dy
            dh = Why' * dy(:, t) + dhnext; %dh = np.dot(Why.T, dy) + dhnext
            
            % this is the most complicated part, need to include dcnext here, since f depends on c(t-1)
            % (1 - c(:, t) .* c(:, t)) - propagate through tanh(c(t))
            dc = (1 - c(:, t) .* c(:, t)) .* (dh .* iofc(K*1+1:K*2, t) + dcnext);
            
            %TODO: possibly this can be optimized/parallelized further
            %gates
            %(o(:, t) .* (1.0 - o(:, t))) - propagate through sigmoid at the same time 
            diofc((K+1):2*K, :) = c(:, t) .* ...
                                (iofc(K*1+1:K*2, t) .* (1.0 - iofc(K*1+1:K*2, t))) .* dh;
            diofc(1:K, :) = iofc(K*3+1:end, t) .* dc .* ...
                                (iofc(1:K, t) .* (1.0 - iofc(1:K, t)));
            diofc(2*K+1:3*K, :) = c(:, t - 1) .* dc .* ...
                                (iofc(K*2+1:K*3, t) .* (1.0 - iofc(K*2+1:K*3, t)));

            %(1 - cc(:, t) .* cc(:, t)) - propagate through tanh at the same time
            diofc((3*K+1):end, :) = (1 - iofc(K*3+1:end, t) .* iofc(K*3+1:end, t)) .* dc .* iofc(1:K, t);

            % the easy part, linear layers
            dU = dU + diofc * h(:, t - 1)';
            dW = dW + diofc * xs(:, t)';
            db = db + diofc;

            %this part is needed for the next iteration since f 
            %depends on c(t-1) and every gate depends on h(t-1)
            dhnext = U' * diofc;
            dcnext = iofc(K*2+1:K*3, t) .* dc;
        end
        
        elapsed = toc;
        
        % debug code, checks gradients - slow!
        if (elapsed > show_every)
            lstm_grad_check;
        end
        
        % clip gradients to some range

        dWhy = clip(dWhy, -5, 5);
        dby = clip(dby, -5, 5);
        
        dU = clip(dU, -5, 5);
        dW = clip(dW, -5, 5);
        db = clip(db, -5, 5);
              
        % % adjust weights, adagrad:
        mWhy = mWhy + dWhy .* dWhy;
        mby = mby + dby .* dby;
        
        mU = mU + dU .* dU;
        mW = mW + dW .* dW;
        mb = mb + db .* db;
        
        Why = Why - learning_rate * dWhy ./ (sqrt(mWhy + eps));
        by = by - learning_rate * dby ./ (sqrt(mby + eps));
        
        U = U - learning_rate * dU ./ (sqrt(mU + eps));
        W = W - learning_rate * dW ./ (sqrt(mW + eps));
        b = b - learning_rate * db ./ (sqrt(mb + eps));
        
        % %%%%%%%%%%%%%%%%%%%%%
        
        smooth_loss = smooth_loss * 0.999 + loss * 0.001;
        
        % show stats every show_every s
        if (elapsed > show_every)
            
            loss_history = [loss_history smooth_loss];
            
            fprintf('[epoch %d] %d %% text read... smooth loss = %.3f\n', e, round(100 * ii / text_length), smooth_loss);
            fprintf('\n\nGenerating some text...\n');
            
            % random h,c seeds
            t = generate_lstm_compact(	U, W, b, Why, by, ...
                500, tanh(zeros(size(h(:, 1)))), tanh(zeros(size(c(:, 1)))));
            %t = generate_rnn(Wxh, Whh, Why, bh, by, 1000, clip(randn(size(Why, 2), 1) * 0.5, -1, 1));
            % generate according to the last seen h
            % t = generate_lstm(	Ui, Wi, bi, ...
            %     Uf, Wf, bf, ...
            %     Uo, Wo, bo, ...
            %     Uc, Wc, bc, ...
            %     Why, by, ...
            %     500, h(:, seq_length), c(:, seq_length));
            fprintf('%s \n', t);
            
            % update plots
            figure(1)
            subplot(2, 5, 1);
            imagesc(W'); 
            hold on;
            plot(linspace(K,K,vocab_size), 1:vocab_size ); 
            plot(linspace(K*2,K*2,vocab_size), 1:vocab_size );
            plot(linspace(K*3,K*3,vocab_size), 1:vocab_size ); 
            hold off;
            title('W (i, o, f, c)');

            subplot(2, 5, 2);
            imagesc(U');
            hold on;
            plot(linspace(K,K,K*4), 1:K*4 ); 
            plot(linspace(K*2,K*2,K*4), 1:K*4 );
            plot(linspace(K*3,K*3,K*4), 1:K*4 ); 
            hold off;
            title('U (i, o, f, c)');

            subplot(2, 5, 6);
            imagesc(dW');
            hold on;
            plot(linspace(K,K,vocab_size), 1:vocab_size ); 
            plot(linspace(K*2,K*2,vocab_size), 1:vocab_size );
            plot(linspace(K*3,K*3,vocab_size), 1:vocab_size ); 
            hold off;
            title('dW (i, o, f, c)');

            subplot(2, 5, 7);
            imagesc(dU');
            hold on;
            plot(linspace(K,K,K*4), 1:K*4 ); 
            plot(linspace(K*2,K*2,K*4), 1:K*4 );
            plot(linspace(K*3,K*3,K*4), 1:K*4 ); 
            hold off;
            title('dU (i, o, f, c)');
 
            subplot(2, 5, 3);
            imagesc(Why');
            title('Why');

            subplot(2, 5, 4);
            imagesc((h + 1) / 2);
            title('h');

            subplot(2, 5, 5);
            imagesc(probs);
            title('probs');

            subplot(2, 5, 10);
            plot(loss_history);
            title('Loss history');
 
        figure(1);
        subplot(2,5,8);
        histogram(h);
   
        subplot(2,5,9);
        histogram(c);
        
            drawnow;
            
            % reset timer
            tic
            
        end
        
        %carry
        iofc(:, 1) = iofc(:, seq_length);
        c(:, 1) = c(:, seq_length);
        tc(:, 1) = c(:, seq_length);
        h(:, 1) = h(:, seq_length);
        y(:, 1) = y(:, seq_length);
        probs(:, 1) = probs(:, seq_length);
        
    end
    
end