import jittor as jt


class MeshTensor:
    """
    A MeshTensor object stores a batch of triangular meshes with 
    multi-dimensional arrays.

    All faces are stored in a 3-dimensional tensor. To support a batch of 
    variable number of faces, an addtional array Fs is used to hold every mesh's
    number of faces. 
    """
    def __init__(self, faces: jt.Var, feats: jt.Var, Fs: jt.Var=None, cache=None):
        """
        Parameters
        ------------
        faces: (N, F, 3) int32
            Array of triangular faces.
        feats: (N, C, F) float32
            Array of face features.
        Fs: (N,) int32, optional
            Array of number of faces in each mesh. 
            If not specified, Fs is set to n.
        cache: dict, optional
            things calculated from faces to avoid repeated calculation for the 
            same mesh.
        """
        self.faces = faces
        self.feats = feats

        self.N, self.C, self.F = feats.shape

        if Fs is not None:
            self.Fs = Fs 
            assert self.F == self.Fs.max().data[0]
        else:
            self.Fs = jt.ones(self.N, dtype="int32") * self.F

        self._cache = cache if cache is not None else {}

    def updated(self, new_feats):
        """ 
        Return a new MeshTensor with its feats updated. 
        
        A shortcut to obtain a new MeshTensor with new features.
        """
        assert new_feats.shape[0] == self.N
        assert new_feats.shape[2] == self.F
        return MeshTensor(self.faces, new_feats, self.Fs, self._cache)

    @property
    def shape(self):
        return self.feats.shape

    @property
    def V(self) -> int:
        """ Maximum number of vertices in the mini-batch """
        if not 'V' in self._cache:
            self._cache['V'] = int((self.faces.max() + 1).data)
        return self._cache['V']

    @property
    def Vs(self) -> jt.Var:
        """ 
        Number of vertices in each mesh. 
        
        Returns
        ------------
        (N,) int32
        """
        if not 'Vs' in self._cache:
            self._cache['Vs'] = self.faces.max(dim=1).max(dim=1) + 1
        return self._cache['Vs']

    @property
    def degrees(self) -> jt.Var:
        """
        Degrees of vertices.

        Return:
        ------------
        (N, V) int32
        """
        if not 'degrees' in self._cache:
            face_degrees = jt.ones((self.N, self.F, 3), dtype=jt.int32)
            self._cache['degrees'] = face_degrees.reindex_reduce(
                op='add',
                shape=[self.N, self.V],
                indexes=[
                    'i0', '@e0(i0, i1, i2)'
                ],
                extras=[self.faces, self.Fs],
                overflow_conditions=['i1 >= @e1(i0)']
            )
        return self._cache['degrees']

    @property
    def FAF(self) -> jt.Var:
        """ 
        FAF (Face-adjacent-faces) indexs the adjacencies.

        Returns:
        ------------
        (N, F, 3) int32
        """
        if not 'FAF' in self._cache:
            self._cache['FAF'] = self.compute_face_adjacency_faces()
        return self._cache['FAF']

    @property
    def FAFP(self) -> jt.Var:
        """ The previous face of current face's adjacent faces """
        if not 'FAFP' in self._cache:
            self._cache['FAFP'], self._cache['FAFN'] = self.compute_face_adjacency_reordered()
        return self._cache['FAFP']

    @property
    def FAFN(self) -> jt.Var:
        """ The next face of current face's adjacent faces """
        if not 'FAFN' in self._cache:
            self._cache['FAFP'], self._cache['FAFN'] = self.compute_face_adjacency_reordered()
        return self._cache['FAFN']

    def __add__(self, other: jt.Var) -> jt.Var:
        new_feats = self.feats + other.feats
        return self.updated(new_feats)

    def __radd__(self, other: jt.Var) -> jt.Var:
        return self.__add__(other)

    def __sub__(self, other: jt.Var) -> jt.Var:
        new_feats = self.feats - other.feats
        return self.updated(new_feats)

    def __rsub__(self, other: jt.Var) -> jt.Var:
        new_feats = other.feats - self.feats
        return self.updated(new_feats)

    def __repr__(self):
        return f'MeshTensor: N={self.N}, C={self.C}, F={self.F}'

    def inverse_loop_pool(self, op='max', pooled_feats=None):
        """ 
        Pooling with the inverse loop scheme.

        Parameters:
        ------------
        op: {'max', 'mean'}, optional
            Reduction method of pooling. The default is 'max'.
        pooled_feats: (N, C, F) float32, optional
            Specifying the feature after pooling.

        Returns:
        ------------
        MeshTensor after 4-to-1 face merge.
        """
        pooled_Fs = self.Fs // 4

        pooled_faces = self.faces.reindex(
            shape=[self.N, self.F // 4, 3],
            indexes=[
                'i0',
                'i1 + @e0(i0) * i2',
                '0',
            ],
            extras=[pooled_Fs],
            overflow_conditions=['i1 >= @e0(i0)'],
            overflow_value=0
        )

        if pooled_feats is None:
            pooled_feats = self.feats.reindex(
                shape=[self.N, self.C, self.F // 4, 4],
                indexes=[
                    'i0',
                    'i1',
                    'i2 + @e0(i0) * i3'
                ],
                extras=[pooled_Fs],
                overflow_conditions=['i2 >= @e0(i0)'],
                overflow_value=0
            )

            if op == 'max':
                pooled_feats = jt.argmax(pooled_feats, dim=3)[1]
            elif op == 'mean':
                pooled_feats = jt.mean(pooled_feats, dim=3)
            else:
                raise Exception('Unsupported pooling operation')
        else:
            assert pooled_feats.shape[0] == self.N
            assert pooled_feats.shape[2] == self.F // 4

        return MeshTensor(pooled_faces, pooled_feats, pooled_Fs)

    def loop_subdivision(self):
        """
        Computes the faces of meshes after one time of loop subdivision.
        """
        subdiv_faces = jt.zeros([self.N, self.F * 4, 3], dtype=jt.float32)
        for i in range(self.N):
            V = self.faces[i].max() + 1
            F = self.Fs[i].data[0]

            E = jt.concat([
                self.faces[i, :F, [0,1]],
                self.faces[i, :F, [1,2]],
                self.faces[i, :F, [2,0]]
            ], dim=0)
            E_hash = E.min(dim=1).astype('int64') * E.max() + E.max(dim=1)
            E2F, _ = jt.argsort(E_hash)
            F2E = jt.zeros_like(E2F)
            F2E[E2F] = jt.index((E.shape[0],), 0) // 2

            E2 = V + F2E[:F]
            E0 = V + F2E[F:F*2]
            E1 = V + F2E[F*2:]
            subdiv_faces[i, :F*4] = jt.concat([
                jt.stack([self.faces[i, :F, 0], E2, E1], dim=-1),
                jt.stack([self.faces[i, :F, 1], E0, E2], dim=-1),
                jt.stack([self.faces[i, :F, 2], E1, E0], dim=-1),
                jt.stack([E0, E1, E2], dim=-1)
            ], dim=0)
        return subdiv_faces

    def loop_unpool(self, mode, ref_faces=None, ref_cache=None):
        """
        Unpooling with the loop subdivision scheme.

        Parameters:
        ------------
        mode: {'nearest', 'bilinear'}
            Algorithm used for unpooling.
        ref_faces: (N, F, 3) int32, optional
            If specified, the returned MeshTensor uses the reference faces 
            instead of computing by loop subdivision. This parameter can speed 
            up dense prediction networks with pairs of pooling and unpooling. 
            The default is None.
        ref_cache: dict, optional
            If specified, the returned MeshTensor uses the reference cache. The
            default is None.

        Returns:
        ------------
        MeshTensor after 1-to-4 face split.
        """
        unpooled_Fs = self.Fs * 4

        if ref_faces is not None:
            unpooled_faces = ref_faces
            unpooled_cache = ref_cache
        else:
            unpooled_faces = self.loop_subdivision()
            unpooled_cache = None

        if mode == 'nearest':
            unpooled_feats = jt.concat([self.feats] * 4, dim=2)
        elif mode == 'bilinear':
            neighbor_feats = self.feats.reindex(
                shape=[self.N, self.C, self.F, 3],
                indexes=[
                    'i0', 'i1', '@e0(i0, i2, i3)'
                ],
                extras=[self.FAF]
            )
            unpooled_feats = jt.concat([
                (self.feats * 2 + neighbor_feats[..., 1] + neighbor_feats[..., 2]) / 4,
                (self.feats * 2 + neighbor_feats[..., 2] + neighbor_feats[..., 0]) / 4,
                (self.feats * 2 + neighbor_feats[..., 0] + neighbor_feats[..., 1]) / 4,
                self.feats
            ], dim=2)
        else:
            raise Exception(f'Unsupported unpool mode: {mode}')

        return MeshTensor(unpooled_faces, unpooled_feats, unpooled_Fs, unpooled_cache)

    def compute_face_adjacency_faces(self) -> jt.Var:
        """ 
        Compute face adjacency faces.

        Returns:
        ------------
        (N, F, 3) int32
        """
        FAF = jt.zeros_like(self.faces)
        for i in range(self.N):
            F = self.Fs[i].data[0]
            E = jt.concat([
                self.faces[i, :F, [1, 2]],
                self.faces[i, :F, [2, 0]],
                self.faces[i, :F, [0, 1]],
            ], dim=0)

            E_hash = E.min(dim=1).astype('int64') * E.max() + E.max(dim=1)

            # S is index of sorted E_hash.
            # Based on the construction rule of E,
            #   1. S % F is the face id
            #   2. S // F is the order of edge in F
            S, _ = jt.argsort(E_hash)

            # S[:, 0] and S[:, 1] are pairs of half-edge
            S = S.reshape(-1, 2)

            FAF[i, S[:, 0] % F, S[:, 0] // F] = S[:, 1] % F
            FAF[i, S[:, 1] % F, S[:, 1] // F] = S[:, 0] % F

        return FAF

    def compute_face_adjacency_reordered(self) -> jt.Var:
        """
        """
        FAF = self.FAF

        FAF_ext = FAF.reindex(
            shape=[self.N, self.F, 3, 3],
            indexes=[
                'i0', '@e0(i0, i1, i2)', 'i3',
            ],
            extras=[FAF],
        )

        # shift adjacency so that
        for _ in range(2):
            FAF_ext = FAF_ext.reindex(
                shape=[self.N, self.F, 3, 3],
                indexes=[
                    'i0', 'i1', 'i2', '@e0(i0, i1, i2, 0) == i1 ? i3 : (i3 > 0 ? i3 - 1 : 2)'
                ],
                extras=[FAF_ext]
            )

        FAFP = FAF_ext[:, :, :, 2]
        FAFN = FAF_ext[:, :, :, 1]
        return FAFP, FAFN

    def dilated_face_adjacencies(self, dilation: int):
        if dilation <= 1:
            raise Exception('dilation must be greater than zero')

        DFA = jt.code(
            shape=[self.N, self.F, 3],
            dtype=jt.int32,
            inputs=[self.FAF, jt.zeros((dilation, 0), dtype=jt.int32)],
            cpu_src="""
                @alias(FAF, in0)
                int dilation = in1_shape0;

                for (int bs = 0; bs < out_shape0; ++bs)
                    for (int f = 0; f < out_shape1; ++f)
                        for (int k = 0; k < out_shape2; ++k) {
                            int a = f;
                            int b = @FAF(bs, f, k);
                            for (int d = 1; d < dilation; ++d) {
                                int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                                a = b;
                                if ((d & 1) == 0) {       // go to next
                                    b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                                } else {                // go to previous
                                    b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                                }
                            }
                            @out(bs, f, k) = b;
                        }
            """,
            cuda_src="""
                __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(FAF, in0)
                    int dilation = in1_shape0;
                    int N = in0_shape0;
                    int F = in0_shape1;

                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int bs = idx / (F * 3);
                    int f = idx / 3 % F;
                    int k = idx % 3;

                    if (bs >= N)
                        return;

                    int a = f;
                    int b = @FAF(bs, f, k);
                    for (int d = 1; d < dilation; ++d) {
                        int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                        a = b;
                        if ((d & 1) == 0) {     // go to next
                            b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                        } else {                // go to previous
                            b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                        }
                    }
                    @out(bs, f, k) = b;
                }

                dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);
            """
        )

        return DFA

    def convolution_kernel_pattern(self, kernel_size=3, dilation=1):
        if kernel_size == 1:
            raise Exception(f'kernel size 1 does not have convolution pattern')

        if kernel_size == 3:
            if dilation == 1:
                return self.FAF
            else:
                return self.dilated_face_adjacencies(dilation)
        elif kernel_size == 5:
            if dilation == 1:
                return jt.stack([
                    self.FAFN[:, :, 0],
                    self.FAF[:, :, 0],
                    self.FAFP[:, :, 0],
                    self.FAFN[:, :, 1],
                    self.FAF[:, :, 1],
                    self.FAFP[:, :, 1],
                    self.FAFN[:, :, 2],
                    self.FAF[:, :, 2],
                    self.FAFP[:, :, 2],
                ], dim=-1)
            else:
                raise Exception('Not support dilation with kernel size larger than 3 yet')
        else:
            DFA = jt.code(
                shape=[self.N, self.F, 3],
                dtype=jt.int32,
                inputs=[self.FAF, jt.zeros(kernel_size, 0), jt.zeros((dilation, 0), dtype=jt.int32)],
                cpu_src="""
                    @alias(FAF, in0)
                    int kernel_size = in1_shape0;
                    int dilation = in2_shape0;

                    for (int bs = 0; bs < out_shape0; ++bs)
                        for (int f = 0; f < out_shape1; ++f)
                            for (int k = 0; k < out_shape2; ++k) {
                                int a = f;
                                int b = @FAF(bs, f, k);
                                for (int d = 1; d < 0; ++d) {
                                    int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                                    a = b;
                                    if ((d & 1) == 0) {       // go to next
                                        b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                                    } else {                // go to previous
                                        b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                                    }
                                }
                                @out(bs, f, k) = b;
                            }
                """,
                cuda_src="""
                    __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                        @PRECALC
                        @alias(FAF, in0)
                        int dilation = in1_shape0;
                        int N = in0_shape0;
                        int F = in0_shape1;

                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        int bs = idx / (F * 3);
                        int f = idx / 3 % F;
                        int k = idx % 3;

                        if (bs >= N)
                            return;

                        int a = f;
                        int b = @FAF(bs, f, k);
                        for (int d = 1; d < dilation; ++d) {
                            int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                            a = b;
                            if ((d & 1) == 0) {     // go to next
                                b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                            } else {                // go to previous
                                b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                            }
                        }
                        @out(bs, f, k) = b;
                    }

                    dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);
                """
            )

            return DFA

            raise Exception(f'Unspported kernel size {kernel_size}')


    def aggregate_vertex_feature(self, op='mean'):
        """
            Aggregate face feature to vertex

            Parameters:
            -----------
                op: {'min', 'max', 'mean'}, optional
                
            Returns:
            --------
                vertex_features: (N, C, V), float32
        """
        if not op in ['max', 'min', 'maximum', 'minimum', 'mean']:
            raise Exception(f'Unsupported op: {op}')
        jt_op = op
        if op == 'max':
            jt_op = 'maximum'
        if op == 'min':
            jt_op = 'minimum'
        if op == 'mean':
            jt_op = 'add'

        face_features = jt.misc.repeat(
            self.feats.unsqueeze(dim=-1),
            [1, 1, 1, 3]
        )
        vertex_features = face_features.reindex_reduce(
            op=jt_op,
            shape=[self.N, self.C, self.V],
            indexes=[
                'i0',
                'i1',
                '@e0(i0, i2, i3)'
            ],
            extras=[self.faces, self.Fs],
            overflow_conditions=['i2 >= @e1(i0)']
        )

        if op == 'mean':
            degree = self.degrees.reindex(
                shape=[self.N, self.V],
                indexes=['i0', 'i1'],
                extras=[self.Vs],
                overflow_value=1,
                overflow_conditions=['i1 >= @e0(i0)']
            )
            vertex_features = vertex_features / degree.unsqueeze(dim=1)

        return vertex_features
