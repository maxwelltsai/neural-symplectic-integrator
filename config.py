CONFIG = {
    'input_dim': 6, # two bodies, each with q_x, q_y, p_z, p_y
    'output_dim': 1, # just one scalar, which is the Hamiltonian
    'hidden_dim': 512,
    'learn_rate': 1e-4,
    'input_noise': 0.,
    'batch_size': 512,
    'activation': 'SymmetricLog',
    'backbone': 'SparseMLP',
    'total_steps': 200,
    'print_every': 200,
    'verbose': True,
    'name': 'wh',
    'seed': 3,
    'device': 'cpu',
    'fig_dir': './figures',
    'data_dir': '.',
    'save_dir': '.',
}
