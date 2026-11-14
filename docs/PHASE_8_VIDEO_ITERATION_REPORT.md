# Báo cáo toàn bộ quá trình xử lý Video — Phase 8 Factory

Thời điểm chốt: **2026-11-14T23:12:19+07:00**. Branch làm việc: `main`. Baseline trước commit báo cáo: `e74f674fbfb6876b7b5899a7b64458801e3a683d`.

## 1. Trả lời thẳng

**Đúng: Video đang là phần khó và tốn nhiều vòng nhất của Phase 8.** Khó khăn đến từ cả tích hợp sai/chưa đủ của Factory, đặc tính backend/checkpoint, giới hạn tài nguyên, chất lượng output và công cụ đánh giá. Không thể quy toàn bộ cho model, cũng không thể coi chạy xong là đã đạt.

Từ hồ sơ hiện có: **32 lượt GPU đã gửi**, gồm **26 lượt native certification** và **6 lượt chẩn đoán**. **31 lượt đã thu receipt**, một lượt đang chờ. Các native case ghi nhận **85 output hoàn chỉnh** (bao gồm warmup, các capability trong cùng batch và các lượt không đạt). Ngoài số này, codec diagnosis và sampler diagnosis có thêm 2 full Transformer calls; lần giải mã FP32 cùng latent chỉ là component decode. Những lần chết trước khi xuất video không được tính thành output.

**Không có số lần sửa code chính xác có thể kiểm toán theo commit**, vì toàn bộ công việc Phase 0–8 tích lũy trong một worktree chưa commit từng vòng. Số phiên notebook, số capability, số video, số lần sửa code và số test case là các đơn vị khác nhau. Các submission bị từ chối trước khi tạo version không nằm trong 32 lượt nói trên. Phụ lục dưới đây liệt kê từng version đã ghi nhận thay vì đoán số lần chỉnh sửa.

Hiện tại: **Wave D 6/13 capability V3; toàn Phase 8 là 32 capability V3; production-ready 0; Exit Gate Phase 8 NOT_MET.** Full Factory E2E V4 thuộc Phase 9, chưa thực hiện.

## 2. Những capability Video đã đạt và chưa đạt

| Model | Capability | Trạng thái hiện tại |
|---|---|---|
| Wan TI2V-5B | T2V | NOT_MET; v8 trượt semantic margin ở một lượt |
| Wan TI2V-5B | I2V | V3 PASS, v1 |
| Wan TI2V-5B | FL2V | V3 PASS, v2; Adapter vẫn là thực nghiệm có khai báo |
| Wan TI2V-5B | Extension | V3 PASS, v1 |
| MiniMax Base | T2V | V3 PASS, v3 |
| MiniMax Base | I2V | NOT_MET; numeric đạt nhưng content fail |
| MiniMax Base | FL2V | NOT_MET; numeric đạt nhưng content fail |
| MiniMax Base | Video + Audio | NOT_MET; v2 đúng thoại nhưng lỗi hình ảnh |
| MiniMax Reference | Reference conditioning | V3 PASS, v1 |
| MiniMax Reference | Video + Audio | V3 PASS, v2 |
| MiniMax Turbo | T2V | NOT_MET; clipping/bố cục và content |
| MiniMax Turbo | I2V | NOT_MET; vùng sáng bất thường |
| MiniMax Turbo | FL2V | NOT_MET; vùng sáng bất thường |

PASS chỉ áp dụng cho exact checkpoint/backend/runtime/golden và bằng chứng đã ghi. Các giới hạn chi tiết/flicker vẫn được ghi trong direct review; không phải bảo đảm mọi prompt đều đạt. Chưa capability nào được công bố production-ready.

## 3. Các nhóm chỉnh sửa và nghiên cứu thực sự đã làm

### Wan

1. Sửa tích hợp `from_pipe` để không âm thầm nâng shared FP16 weights thành FP32; xác thực sharing và dtype thực.
2. Tách VAE FP32 theo nguồn checkpoint gốc, giữ diffusion/text FP16.
3. Khai báo và xác thực các tham số Transformer/text encoder cần giữ FP32 thay vì ép đồng loạt FP16.
4. Chặn geometry ngoài lattice 32 pixel; không âm thầm đổi kích thước.
5. Xử lý host cgroup OOM bằng sequential CPU offload được khai báo trong Manifest; không fallback tự động.
6. Xử lý VAE decode CUDA OOM bằng tiling/stride tĩnh đã khai báo; đo lại trên GPU thật.
7. Chuyển golden sang native spatial/temporal context, giữ checkpoint và ghi rõ version; chạy lại đủ seed.
8. Chẩn đoán CPU với VAE gốc xác nhận backend TI2V-5B đã pin bỏ qua `last_image`; fail-closed đường FL2V thường.
9. Theo lựa chọn của người dùng, giữ nguyên 5B và nghiên cứu Adapter thực nghiệm: joint context hai endpoint, chỉ giữ mask ở đầu/cuối, Transformer sinh phần giữa; không dán/crossfade RGB output.
10. Bổ sung source hashes, input/mask/latent/seed/MP4 bindings cho từng lượt FL2V. V1 bị gate loại vì root logger lọc INFO; sửa stdout flush và chạy lại v2, không dựng lại log lịch sử.
11. Điều chỉnh prompt T2V có khai báo để làm rõ hướng phải và giữ bóng trong khung; v8 tiến bộ nhưng chưa vượt toàn bộ oracle.
12. Phát hiện giới hạn center crop của CLIP: bóng cuối ở x≈1206, trong khi vùng nhìn giữa của verifier trên canvas1280×704 xấp xỉ288..992. Đây là giả thuyết về phép đo, không được dùng để hạ ngưỡng hoặc tự cấp PASS.

### MiniMax Base / Reference / Turbo

1. Sửa schema reference Autogrow sang dependency links phẳng `ref_images.ref_image_N`; chặn keyframe xung đột capability.
2. Khai báo compute dtype rõ ràng, private loader có attestation runtime; phân biệt định dạng lượng tử hóa của file với dtype tính toán. Precision của các lượt đầu không đạt; cấu hình hiện tại diffusion FP32 được kiểm tra thực.
3. Sửa normalize `SaveVideo`: metadata native nằm ở `images` nhưng artifact thực là MP4; chỉ chấp nhận output node/format đúng.
4. Sửa preflight frame envelope và version golden; sau đó chẩn đoán miền 124 frames để kiểm tra giả thuyết temporal range. 124 frames vẫn không sửa được các vùng sáng bất thường.
5. Đưa deadline vào Manifest theo số liệu chạy thật; không retry model hoặc đổi checkpoint tự động.
6. Phân biệt reserve VRAM với DynamicVRAM extra headroom; Turbo đi qua nhiều vòng residency trước khi v4 hoàn tất 12 video với headroom0.75GiB.
7. Sửa fixture resolver cho list/dict để Reference nhận đúng bytes thay vì chuỗi `@asset`.
8. Dùng biểu diễn native dialogue đã khai báo để sửa nội dung thoại: Base AV v2 và Reference AV v2 đều đúng4/4 câu; Base vẫn fail hình ảnh.
9. Chẩn đoán cùng sampled latent với VAE FP16/FP32: giữ cùng bytes VAE, input và latent; FP32 không loại lỗi. Không đổi canonical precision theo suy đoán.
10. Đối chiếu workflow chính thức và thử riêng sampler `res_multistep`; vẫn lỗi. Không đổi canonical sampler theo một giả thuyết âm tính.
11. Thử canvas1152×768 trong profile chẩn đoán riêng (deadline/headroom có khai báo); các lượt đầu ngắt trước output. Không tăng giới hạn production hoặc nhận V3 từ chẩn đoán.
12. Bổ sung dữ liệu trước shutdown, quan sát exit race và từng health subcheck. V3 ghi nhận SIGTERM(-15), cgroup OOM counters0. Supervisor có đường dừng backend sau health fail; exact subcheck vẫn đang xác minh ở v4. Không khẳng định OOM hay health timeout là nguyên nhân khi chưa có bằng chứng.

### Harness và gate

- Giới hạn payload Kaggle dưới1MB, chỉ bundle đúng fixtures/batch và tách các capability nặng.
- Xem toàn bộ khung hình của mỗi bộ bốn output; numeric PASS bị từ chối khi có lỗi hình ảnh thực. Không chọn seed sạch để cấp PASS.
- Phát hiện khác biệt FFmpeg macOS/Linux làm PCM hash khác; giữ phép so sánh byte-exact và kiểm toán trên decoder Linux đã pin.
- Chạy5 phiên kiểm toán CPU. V1–4 chưa đủ để nhận làm gate tổng; v5 kiểm tra đủ10/10 receipt gốc, exact decoder/PCM/source/golden/review và đã được nhận vào gate.
- Sao chép byte-identical hai archive failed-run sang dataset riêng tư cùng tài khoản để khắc phục Kaggle không mount chúng. Đây là vận chuyển bằng chứng cũ, không phải inference mới.
- Chạy lại source-scoped regression các phase trước khi thay core. Source hiện tại `7266e8243bfef0b0161a8e4711619893f4263fd770894f802d13151fdd613f94` có snapshot gates4–7 PASS; các sửa chẩn đoán mới chỉ nằm trong benchmark/deployment tooling.

## 4. Nhật ký đầy đủ các lượt GPU đã ghi nhận

`complete/error` là trạng thái job Kaggle. `error` có thể do golden không đạt dù đã tạo đủ video, không mặc định là crash. Numeric PASS trong wave_d.json cũng không thay thế content review và strict V3 gate.

| Batch | Version | Kaggle | Output native hoàn chỉnh | Kết luận / điều chỉnh |
|---|---:|---|---:|---|
| `wan_t2v` | 1 | error | 0 | Mất tiến trình sau load; thiếu số liệu để kết luận OOM. Phát hiện from_pipe mặc định làm rộng shared weights. |
| `wan_t2v` | 2 | error | 4 | Có 4 video; lỗi hướng/hiển thị và tái dựng VAE, chưa đạt. |
| `wan_t2v` | 3 | error | 0 | Không có case hoàn chỉnh; nghiên cứu precision/protected parameters. |
| `wan_t2v` | 4 | error | 0 | Hoàn tất denoising nhưng host cgroup OOM trong giai đoạn sau; không có video hoàn chỉnh. |
| `wan_t2v` | 5 | error | 4 | Có 4 video; vẫn chưa đạt chất lượng/numeric oracle. |
| `wan_t2v` | 6 | error | 0 | CUDA OOM ở VAE decode; không có output hoàn chỉnh. |
| `wan_t2v` | 7 | error | 4 | 4 video native grid; hướng/chuyển động/khung hình chưa đạt. |
| `wan_t2v` | 8 | error | 4 | 4 video; cả bốn đạt hướng phải và thấy bóng, lượt 3 trượt text margin cuối (0.022844 < 0.03). |
| `wan_i2v` | 1 | complete | 4 | V3 PASS: đủ 4 lượt và 164 khung hình được xem. |
| `wan_fl2v` | 1 | complete | 4 | Video/numeric/content đạt nhưng thiếu log binding từng lượt; strict gate từ chối. |
| `wan_fl2v` | 2 | complete | 4 | V3 PASS sau sửa transport log; đủ 4 lượt/164 khung hình và endpoint bindings. |
| `wan_extension` | 1 | complete | 4 | V3 PASS: 4 lượt/356 khung hình, có prefix thật và đoạn sinh tiếp. |
| `minimax_t2v` | 1 | error | 0 | Preflight từ chối 124 frames vượt giới hạn Manifest 107; không có inference. |
| `minimax_t2v` | 2 | error | 0 | Chưa có output hoàn chỉnh; precision/normalization integration chưa đạt. |
| `minimax_t2v` | 3 | complete | 4 | V3 PASS: exact checkpoint, FP32 diffusion, 4 lượt/428 khung hình. |
| `minimax_conditioned` | 1 | complete | 8 | 8 video I2V/FL2V numeric PASS; content review loại vì vùng sáng bất thường. |
| `minimax_conditioned` | 2 | complete | 8 | 8 video 124 frames numeric PASS; xem đủ 992 frames vẫn thấy vùng sáng che mặt; cả hai cap chưa đạt. |
| `minimax_turbo` | 1 | error | 0 | Dừng vì giới hạn resident GPU trước sampling. |
| `minimax_turbo` | 2 | error | 0 | Chưa có case hoàn chỉnh; điều chỉnh static headroom có khai báo. |
| `minimax_turbo` | 3 | error | 1 | Chỉ hoàn tất warmup đầu rồi vượt residency ở request tiếp; không đạt. |
| `minimax_turbo` | 4 | error | 12 | Đủ 12 video, residency ổn; T2V có bóng bị cắt, I2V/FL2V có vùng sáng bất thường. Không cap nào V3. |
| `minimax_reference` | 1 | complete | 4 | V3 PASS: đủ 4 video/428 frames, có giới hạn flicker/chi tiết nhỏ. |
| `minimax_audio` | 1 | error | 4 | 4 video có âm thanh; 3/4 lượt trượt câu thoại. |
| `minimax_audio` | 2 | complete | 4 | 4/4 câu thoại đúng, numeric PASS; content FAIL do vùng sáng che mặt ở một lượt. |
| `minimax_reference_audio` | 1 | error | 4 | 4/4 lượt trượt câu thoại do từ thừa. |
| `minimax_reference_audio` | 2 | complete | 4 | V3 PASS sau native-dialogue prompt: 4/4 transcript chính xác, 428 frames content PASS; Linux audit v5 đầy đủ. |
| `minimax_vae_diag` | 1 | complete | chẩn đoán | 1 full Transformer call + cùng latent giải mã FP32: lỗi vẫn còn; giả thuyết chỉ do độ chính xác decoder không được ủng hộ. |
| `minimax_sampler_diag` | 1 | complete | chẩn đoán | 1 full call đổi riêng Euler → res_multistep; vẫn có vùng sáng che mặt, chưa chứng minh đây là cách sửa. |
| `minimax_native_canvas_diag` | 1 | error | chẩn đoán | 1152×768: reset kết nối ở đầu sampling, không latent/MP4. |
| `minimax_native_canvas_diag` | 2 | error | chẩn đoán | Lặp cấu hình, thêm snapshot trước stop; vẫn reset, poll tức thời chưa có exit code, OOM counters 0. |
| `minimax_native_canvas_diag` | 3 | error | chẩn đoán | Quan sát thụ động sau lỗi: SIGTERM(-15) sau khoảng 2.5s; chưa xác định chính xác bước health thất bại. |
| `minimax_native_canvas_diag` | 4 | queued (last monitor) | chẩn đoán | Thêm quan sát _json/listener/GPU ownership, giữ nguyên kết quả các check; đang queued ở snapshot báo cáo. |

Các bảng `PASS` ở raw numeric case của conditioned/Turbo/Audio không phải V3 PASS cuối. File gate đã nhận và direct visual reviews là căn cứ hiện tại.

## 5. Kết quả và giới hạn kiểm chứng

- Bộ phần mềm mặc định gần nhất trước các bổ sung chẩn đoán/kiểm toán cuối: **1447 tests PASS**. Không nhận đây là bằng chứng inference hay tuyên bố đã chạy lại toàn bộ suite sau từng thay đổi cuối.
- Các kiểm tra đóng gói diagnostic gần nhất:5 PASS; các kiểm tra integrity của audit tooling:4 PASS. Đây là software checks.
- Native proof gồm checkpoint hashes, backend identity, GPU/host telemetry, input/output hashes, seed, semantic evaluation, direct content review và lifecycle; cần đọc đủ chuỗi.
- Có thiếu sót quan sát ở lượt đầu (không có exit reason đầy đủ); các kết luận không đủ bằng chứng được ghi là chưa xác định. Chẩn đoán bổ sung mới thu SIGTERM, chưa đủ xác định exact health subcheck.
- Nhiều vòng đã giải quyết được residency/tích hợp/thoại nhưng chưa giải quyết chất lượng của mọi capability. Không có lý do coi Phase8 hoàn tất hay chuyển READY.

## 6. Còn lại và thứ tự tiếp tục

1. Thu native-canvas v4, xác định exact health subcheck trước khi sửa lifecycle/timeout; nếu sửa core phải lặp source regressions và native goldens liên quan.
2. Xử lý MiniMax Base/Turbo I2V/FL2V và Base AV: các lỗi hình ảnh còn thực, các giả thuyết VAE precision và sampler riêng lẻ không giải quyết được case.
3. Với Wan T2V, nếu sửa phép đo phải lập calibration độc lập và policy có version trước khi đánh giá lại; không hạ margin0.03, chọn seed hay tự bỏ case.
4. Chỉ sau WaveD gate mới hoàn tất WaveE: ACE-Step28 capability trong phạm vi hiện tại, pin runtime/Adapter, chặn tải weights ngoài Manifest và chạy GPU thật. YuE giữ BLOCKED có lý do, không giả lập output.
5. WaveF là Enhancement: hiện không có capability enabled; cần chốt truthful scope/gate.
6. Chốt gate tổng Phase8 theo Master Plan. Sau đó Phase9 Full Factory E2E qua API/OpenAI SDK, rồi các phase tiếp theo. V3 không tự thành production-ready.

## 7. Phạm vi đưa lên GitHub

Commit đi kèm báo cáo chứa toàn bộ code, Manifest, cấu hình, tests, tài liệu và bằng chứng dạng file đang được Git cho phép trong worktree Phase0–8; không tự giới hạn chỉ các file Video. Giữ nguyên baseline Master Plan và các quy tắc `.gitignore`: không force-add weights, môi trường ảo, runtime secrets, cache, log/media được ignore. Test fixtures được whitelist vẫn theo dõi bình thường. Vì vậy đây không phải bản sao nhị phân mọi artifact Kaggle; receipt/hash và nguồn lưu vẫn được giữ để đối chiếu. Không có file ứng viên vượt100MB ở thời điểm kiểm tra; tổng dung lượng thô lớn chủ yếu do JSON telemetry/output.

Kiểm tra `git diff --cached --check` báo whitespace trong một số file hiện có và nhiều bản sao nguồn upstream/fixture. Không sửa các snapshot byte-identical chỉ để làm sạch whitespace vì sẽ phá hash bằng chứng. Đây không phải test inference hay lỗi nội dung report.

Hash của artifact không thay thế việc có bytes gốc khi tái kiểm toán. Dữ liệu chuyển riêng tư phục vụ kiểm toán vẫn trên tài khoản Kaggle; không công khai weights hay credentials.

## 8. Nguồn bằng chứng trong repository

- [Master Plan](MASTER_PLAN.md), [findings theo diễn biến](PLAN_FINDINGS.md), [trạng thái nâng cấp](UPGRADE_STATUS.md).
- [Gate Wave D hiện tại](../validation/native_wave_d_gate.json), [provenance kiểm toán đã nhận](../validation/native_wave_d_gate_provenance.json).
- [Review trực tiếp](../validation/native_video_content_review.json), [matrix](../validation/native_matrix.md), [phase status](../validation/phase_status.json).
- [Kiểm toán v5](../validation/native_gate_audit_v5/audit.json), [identity vận chuyển bằng chứng](../validation/diagnostics/native_evidence_transfer/identity.json).
- [Inventory machine-readable của báo cáo](../validation/diagnostics/video_phase8_report_inventory.json). Mỗi dòng chỉ rõ run record/receipt/source bundle và số output; đối chiếu raw observation khi cần.

Không có số liệu suy diễn từ mock/smoke/contract test trong số lượt GPU hoặc capability V3 ở báo cáo này.
