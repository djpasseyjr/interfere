"""Tests for the restricted SINDy module.
These tests check that the restricted SINDy module correctly converts
between adjacency matrices and feature masks, and that it correctly
handles bad input.
"""
import numpy as np
import pysindy as ps
import pytest

from interfere._methods.restricted_sindy import (
    pcmci_graph_to_adjacency_matrix,
    variable_adjacency_matrix_to_feature_mask,
    FeatureMaskOptimizer
)



SAMPLE_PCMCI_GRAPH_TO_ADJ = [
    # Test case 1.
    [
        # PCMCI graph.
        np.array([[['', '-->'],
        ['o-o', '-->'],
        ['', ''],
        ['', ''],
        ['', ''],
        ['', ''],
        ['', ''],
        ['', '']],

       [['o-o', ''],
        ['', '-->'],
        ['', ''],
        ['-->', '-->'],
        ['', '-->'],
        ['', ''],
        ['', ''],
        ['o-o', '-->']],

       [['', ''],
        ['', ''],
        ['', '-->'],
        ['', ''],
        ['', ''],
        ['', ''],
        ['', ''],
        ['x-x', '']],

       [['', ''],
        ['<--', ''],
        ['', ''],
        ['', '-->'],
        ['o-o', ''],
        ['<--', ''],
        ['', ''],
        ['', '-->']],

       [['', ''],
        ['', ''],
        ['', ''],
        ['o-o', ''],
        ['', '-->'],
        ['<--', '-->'],
        ['', '-->'],
        ['', '']],

       [['', ''],
        ['', ''],
        ['', ''],
        ['-->', ''],
        ['-->', ''],
        ['', '-->'],
        ['x-x', ''],
        ['-->', '']],

       [['', ''],
        ['', ''],
        ['', ''],
        ['', ''],
        ['', ''],
        ['x-x', ''],
        ['', '-->'],
        ['', '']],

       [['', ''],
        ['o-o', '-->'],
        ['x-x', ''],
        ['', ''],
        ['', ''],
        ['<--', ''],
        ['', ''],
        ['', '-->']]], dtype='<U3'),

        # Adj matrix of lag 1.
        np.array([
            [ True, False, False, False, False, False, False, False],
            [ True,  True, False, False, False, False, False,  True],
            [False, False,  True, False, False, False, False, False],
            [False,  True, False,  True, False, False, False, False],
            [False,  True, False, False,  True, False, False, False],
            [False, False, False, False,  True,  True, False, False],
            [False, False, False, False,  True, False,  True, False],
            [False,  True, False,  True, False, False, False,  True]
        ])
    ]
]

SAMPLE_ADJ_TO_FEAT_MASK_DATA = [
    # Test case 1.
    [
        # Variable names.
        ["x1", "x2"],
        # Feature names.
        ["x1", "x2"],
        # Adj matrix.
        np.array([
            [True, False],
            [False, True]
        ]),
        # Feature matrix.
        np.array([
            [True, False],
            [False, True]
        ])
    ],

    # Test case 2.
    [
        # Variable names.
        ["x1", "x2"],
        # Feature names.
        ["x1^2", "x2^2"],
        # Adj matrix.
        np.array([
            [True, False],
            [False, True]
        ]),
        # Feature matrix.
        np.array([
            [True, False],
            [False, True]
        ])
    ],

    # Test case 3.
    [
        # Variable names.
        ["x1", "x2"],
        # Feature names.
        ["x1", "x2", "x1^2", "x2^2"],
        # Adj matrix.
        np.array([
            [True, False],
            [False, True]
        ]),
        # Feature matrix.
        np.array([
            [True, False, True, False],
            [False, True, False, True]
        ])
    ],

    # Test case 4.
    [
        # Variable names.
        ["x1", "x2"],
        # Feature names.
        ["1", "x2", "x1^2", "x2^2"],
        # Adj matrix.
        np.array([
            [True, False],
            [False, True]
        ]),
        # Feature matrix.
        np.array([
            [True, False, True, False],
            [True, True, False, True]
        ])
    ],

    # Test Case 5. Real use case.
    [
        # Variable names.
        [f"x{i}" for i in range(7)] + ["u0"],
        # Feature names.
        [
            '1', 'x0', 'x1',                'x2', 'x3', 'x4',               'x5', 'x6', 'u0',               'x0^2', 
            'x0 x1', 'x0 x2', 'x0 x3',      'x0 x4', 'x0 x5', 'x0 x6',      'x0 u0', 'x1^2', 'x1 x2',       'x1 x3', 
            'x1 x4', 'x1 x5', 'x1 x6',      'x1 u0', 'x2^2', 'x2 x3',       'x2 x4', 'x2 x5', 'x2 x6',      'x2 u0', 
            'x3^2', 'x3 x4', 'x3 x5',       'x3 x6', 'x3 u0', 'x4^2',       'x4 x5', 'x4 x6', 'x4 u0',      'x5^2', 
            'x5 x6', 'x5 u0', 'x6^2',       'x6 u0', 'u0^2'
        ],
        # Adj matrix.
        np.array([
            [ True, False, False, False, False, False, False, False],
            [ True,  True, False, False, False, False, False,  True],
            [False, False,  True, False, False, False, False, False],
            [False,  True, False,  True, False, False, False, False],
            [False,  True, False, False,  True, False, False, False],
            [False, False, False, False,  True,  True, False, False],
            [False, False, False, False,  True, False,  True, False],

            [False,  True, False,  True, False, False, False,  True]
        ]),
        # Feature matrix.
        np.array([
            [True if i in [0, 1, 9] else False for i in range(45)],
            [True if i in [0, 2, 17, 1, 9, 8, 44, 10, 16, 23] else False for i in range(45)],
            [True if i in [0, 3, 24] else False for i in range(45)],
            [True if i in [0, 4, 30, 2, 17, 19] else False for i in range(45)],
            [True if i in [0, 5, 35, 2, 17, 20] else False for i in range(45)],
            [True if i in [0, 6, 39, 5, 35, 36] else False for i in range(45)],
            [True if i in [0, 7, 42, 5, 35, 37] else False for i in range(45)],
            [True if i in [0, 8, 44, 2, 17, 4, 30, 19, 23, 34] else False for i in range(45)],
            
        ])
    ]

]

@pytest.mark.parametrize("pcmci_graph, expected_adj", SAMPLE_PCMCI_GRAPH_TO_ADJ)
def test_pcmci_graph_to_adjacency_matrix(pcmci_graph, expected_adj):
    """Test that pcmci graph is correctly converted to adjacency matrix."""
    actual = pcmci_graph_to_adjacency_matrix(pcmci_graph, lag=1)
    assert np.all(actual == expected_adj), (
        f"Actual: \n{actual}\n\nExpected: \n{expected_adj}"
    )


def test_pcmci_graph_to_adjacency_matrix_bad_lag():
    """Test that bad lag values raise value errors."""
    adj = np.ones((2, 2), bool)
    test_pc_graph = np.zeros((2, 2, 1), dtype='<U10')
    test_pc_graph[adj] = '-->'
    with pytest.raises(ValueError):
        pcmci_graph_to_adjacency_matrix(test_pc_graph, lag=10)


def test_pcmci_graph_to_adjacency_matrix_bad_pcmci_graph():
    """Test that bad pcmci graph raises value errors."""
    adj = np.ones((2, 2), bool)
    test_pc_graph = np.zeros((2, 2, 2), dtype='<U10')
    test_pc_graph[adj, 0] = 'o--o'
    with pytest.raises(ValueError):
        pcmci_graph_to_adjacency_matrix(test_pc_graph, lag=0)




@pytest.mark.parametrize(
    "var_names, feat_names, adj, expected",
    SAMPLE_ADJ_TO_FEAT_MASK_DATA,
)
def test_adj_to_feat_mask(var_names, feat_names, adj, expected):
    """Test that variable adj matrix is correctly converted to feature mask.
    """
    actual = variable_adjacency_matrix_to_feature_mask(var_names, feat_names, adj)
    differences = actual != expected
    for i, d in enumerate(differences):
        allowed_vars = [var for var, a in zip(var_names, adj[i, :]) if a]
        actual_feats = [feat for feat, a in zip(feat_names, actual[i, :]) if a]
        exp_feats = [feat for feat, a in zip(feat_names, expected[i, :]) if a]
        assert not np.any(d), (
            f"Adj Matrix allows only: \n\t{allowed_vars}"
            f"\n\nActual Features: \n\t{actual_feats}"
            f"\n\nExpected Features: \n\t{exp_feats}"
        )
    

def test_adj_to_feat_mask_bad_var_name():
    """Test that bad variable names raise value errors."""
    with pytest.raises(ValueError):
        variable_adjacency_matrix_to_feature_mask(["%"], [], [[]])
    
    with pytest.raises(ValueError):
        variable_adjacency_matrix_to_feature_mask([" "], [], [[]])

    with pytest.raises(ValueError):
        variable_adjacency_matrix_to_feature_mask(["x1^2"], [], [[]])

    with pytest.raises(ValueError):
        variable_adjacency_matrix_to_feature_mask(["x1,x2"], [], [[]])


# Test that the feature mask optimizer can force SINDY to make constant 
# derivative predictions.

def test_feat_mask_linear():
    """Test that the feature mask optimizer can force SINDY to make linear 
    predictions.
    """
    t = np.linspace(0, 3, 301)
    X = np.vstack([t, t**2, t**3]).T
    t_test = np.linspace(3, 4, 101)
    sindy = ps.SINDy()
    sindy.fit(X, t)
    X_test_pred = sindy.simulate(X[-1, :], t_test)

    # Check that standard sindy predicts non_constant change.
    change = np.diff(X_test_pred[:, -1])
    assert not np.allclose(change, change[0])

    # Now use feature mask sindy
    mask_sindy = ps.SINDy(
        optimizer=FeatureMaskOptimizer(
            optimizer=ps.STLSQ(),
            feature_mask=np.array([
                [True] + [False] * 9 for i in range(3)
            ])
        )
    )

    mask_sindy.fit(X, t)
    mask_X_test_pred = mask_sindy.simulate(X[-1, :], t_test)

    # If the prediction is linear, all diffs should be equal.
    mask_change = np.diff(mask_X_test_pred[:, -1])
    assert np.allclose(mask_change, mask_change[0]), (
        "Feature optimizer should enforce linear relationship:"
        f"\n\tPredicted (First 5): {mask_X_test_pred[:5, -1]}"
        f"\n\tPredicted (Last 5): {mask_X_test_pred[-5:, -1]}"
        f"\n\tDiff (First 5): {np.diff(mask_X_test_pred[:5, -1])}"
        f"\n\tDiff (Last 5): {np.diff(mask_X_test_pred[-5:, -1])}"
    )


def test_feat_mask_const():
    """Test that the feature mask optimizer can force SINDY to make constant 
    predictions.
    """
    t = np.linspace(0, 3, 301)
    X = np.vstack([t, t**2, t**3]).T
    t_test = np.linspace(3, 4, 101)
    sindy = ps.SINDy()
    sindy.fit(X, t)
    X_test_pred = sindy.simulate(X[-1, :], t_test)

    # Check that standard sindy predicts non_constant change.
    change = np.diff(X_test_pred[:, -1])
    assert not np.allclose(change, change[0])

    # Now use feature mask sindy
    mask_sindy = ps.SINDy(
        optimizer=FeatureMaskOptimizer(
            optimizer=ps.STLSQ(),
            feature_mask=np.array([
                [False] * 10 for i in range(3)
            ])
        )
    )

    mask_sindy.fit(X, t)
    mask_X_test_pred = mask_sindy.simulate(X[-1, :], t_test)

    # If the prediction is constant, then all diffs should be zero.
    mask_change = np.diff(mask_X_test_pred[:, -1])
    assert np.allclose(mask_change, 0), (
        "Feature optimizer should enforce constant state evolution:"
        f"\n\tPredicted (First 5): {mask_X_test_pred[:5, -1]}"
        f"\n\tPredicted (Last 5): {mask_X_test_pred[-5:, -1]}"
        f"\n\tDiff (First 5): {np.diff(mask_X_test_pred[:5, -1])}"
        f"\n\tDiff (Last 5): {np.diff(mask_X_test_pred[-5:, -1])}"
    )


def test_feat_mask_mix():
    """Test that the feature mask optimizer can force SINDY to make linear 
    predictions for some variables and constant predictions for others.
    """
    t = np.linspace(0, 3, 301)
    X = np.vstack([t, t**2, t**3]).T
    t_test = np.linspace(3, 4, 101)

    mask_sindy = ps.SINDy(
        optimizer=FeatureMaskOptimizer(
            optimizer=ps.STLSQ(),
            feature_mask=np.array([
                [False] * 1 + [True] * 9,
                [False] * 10,
                [True] + [False] * 9
            ])
        )
    )

    mask_sindy.fit(X, t)
    mask_X_test_pred = mask_sindy.simulate(X[-1, :], t_test)

    # Should be constant.
    mask_change1 = np.diff(mask_X_test_pred[:, 1])
    assert np.allclose(mask_change1, 0), (
        "Feature optimizer should enforce constant state evolution:"
        f"\n\tPredicted (First 5): {mask_X_test_pred[:5, 1]}"
        f"\n\tPredicted (Last 5): {mask_X_test_pred[-5:, 1]}"
        f"\n\tDiff (First 5): {np.diff(mask_X_test_pred[:5, 1])}"
        f"\n\tDiff (Last 5): {np.diff(mask_X_test_pred[-5:, 1])}"
    )
    
    # Should be linear.
    mask_change2 = np.diff(mask_X_test_pred[:, 2])
    assert np.allclose(mask_change2, mask_change2[0]), (
        "Feature optimizer should enforce linear relationship:"
        f"\n\tPredicted (First 5): {mask_X_test_pred[:5, 2]}"
        f"\n\tPredicted (Last 5): {mask_X_test_pred[-5:, 2]}"
        f"\n\tDiff (First 5): {np.diff(mask_X_test_pred[:5, 2])}"
        f"\n\tDiff (Last 5): {np.diff(mask_X_test_pred[-5:, 2])}"
    )

    # Should be constant.
    assert not np.allclose(mask_change2, 0), (
        "Feature optimizer should enforce linear relationship, got constant:"
        f"\n\tPredicted (First 5): {mask_X_test_pred[:5, 2]}"
        f"\n\tPredicted (Last 5): {mask_X_test_pred[-5:, 2]}"
        f"\n\tDiff (First 5): {np.diff(mask_X_test_pred[:5, 2])}"
        f"\n\tDiff (Last 5): {np.diff(mask_X_test_pred[-5:, 2])}"
    )