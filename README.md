# LiveGuard AI — HID Hackathon

Run this kiosk on the laptop connected to the demo webcam or an external mobile camera. The simulated door checks for liveness, compares a face to the claimed Campus ID, and checks a signed access schedule. It runs locally after dependencies and models are installed.

## Run locally

Use Python 3.12 and a current Chromium or Chrome browser. Commands below use Bash on Linux or macOS. On Windows, use a local Python 3.12 install with PowerShell environment-variable syntax, and install `tzdata` if the system lacks IANA timezone data.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python models.py
python -m unittest discover -s tests -v

# Authorize initial demo subject(s). Use --merge to add subjects without replacing existing grants:
python policy.py --signer-dir state/demo-signer --subject demo-user --merge

# Configure Operator authentication (single operator or JSON mapping)
export LIVEGUARD_OPERATOR_ID="admin-op-01"
export LIVEGUARD_OPERATOR_TOKEN="operator-secret-token-32bytes-secure"

# Start the kiosk server
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). Start the camera, enter `demo-user`, expand **Operator enrollment**, enter the Operator ID and token, confirm consent, enter the participant's display name (supports spaces and Unicode, e.g., Kannada `ಮುರಳಿ`), and click **Enroll campus ID**. Center the face with even lighting and click **Verify access**.

### Preserving Multiple Authorized Subjects

Use the `--merge` flag to add users to the active signed policy snapshot without revoking other enrolled subjects:

```bash
# Add another student/staff member while keeping demo-user authorized:
python policy.py --signer-dir state/demo-signer --subject ddemo-user --merge \
  --timezone Asia/Kolkata --start-minute 0 --end-minute 1440 --ttl 28800

# To revoke all subjects and demonstrate explicit schedule denial:
python policy.py --signer-dir state/demo-signer
```

## Using a Mobile Phone as Camera Input

If the laptop built-in webcam is low quality, broken, or misaligned, connect an Android smartphone:

1. **Android 14+ Native Webcam (USB-C)**:
   - Connect phone via USB-C.
   - On phone: Notification shade > **USB Preferences** > select **Webcam**.
   - In LiveGuard UI: Click **Refresh**, select the Android camera from the dropdown, and click **Start Camera**.
2. **DroidCam over USB (v4l2loopback)**:
   - On Android: Enable USB Debugging in Developer Options and open DroidCam.
   - On Linux: `adb forward tcp:4747 tcp:4747 && droidcam-cli adb 4747`
   - Select `/dev/video2` (or DroidCam virtual camera) in the kiosk UI.
3. **Iriun Webcam**:
   - Install Iriun on phone and Linux laptop (`iriunwebcam`).
   - Automatically establishes USB or local Wi-Fi virtual webcam `/dev/video4`.

## Measure the HID demo

1. Capture consenting volunteers and physical print and screen presentations through the demo webcam. Show the live enrolled person, the wrong person claiming the same ID, poor lighting, and a guided retake. Use the same enrolled ID for print and screen trials to test whether PAD protects a potentially matching identity.
2. Open the **Operator Diagnostics Drawer** on the kiosk to inspect per-frame PAD confidence scores (MiniFASNet V2 and V1SE), SFace cosine similarity matches, focus/blur metrics, and stage timings in real-time.
3. In **Evaluation**, set the scenario before each attempt, then export the session CSV. Export data persists across reloads via browser `sessionStorage` and includes spreadsheet formula injection escaping.
4. Run `python evaluate.py liveguard-session.csv`. It reports browser p50/p95/max, backend p95 where present, deadline misses (against true 2000.0 ms target), PAD outcomes, uncertain results, and attack acceptance.
5. Disconnect the network after setup and demonstrate an offline valid signed policy. Issue a one-second policy with `--ttl 1` to demonstrate expiry, then restore the regular policy. Sync queued audit events after reconnecting.

## AWS sandbox

Use the organizer-provided account, credentials, and allowed region. The workstation needs AWS CLI and AWS SAM CLI. The sandbox must allow CloudFormation/SAM, deployment artifact storage, API Gateway, Lambda, DynamoDB, and IAM role creation or a suitable preapproved role.

```bash
export AWS_REGION='YOUR_ALLOWED_REGION'
aws sts get-caller-identity
sam validate --lint --template-file aws/template.yaml --region "$AWS_REGION"
sam build --template-file aws/template.yaml
sam deploy --guided --stack-name liveguard-hid --region "$AWS_REGION" --capabilities CAPABILITY_IAM
```

During guided deployment, use a sandbox-approved artifact bucket or SAM-managed bucket if allowed. Use the stack's **AuditUrl** output.

```bash
python sync_audit.py --url 'AUDIT_URL_FROM_STACK_OUTPUT' --region "$AWS_REGION"
```

Replaying the same event ID yields one DynamoDB row (idempotent hash deduplication). The kiosk's sync command features exponential backoff retries across network drops. Seven-day expiration is applied by DynamoDB TTL. The AWS deployment contains the audit service and uses no webcam data or identity templates. The kiosk UI and inference run entirely locally.

## Models and Provenance

`python models.py` downloads SHA-256-verified YuNet, SFace, and two MiniFASNet variants. Upstream licenses:
- [YuNet's MIT notice](https://huggingface.co/opencv/face_detection_yunet/blob/main/LICENSE)
- [SFace's Apache-2.0 license](https://huggingface.co/opencv/face_recognition_sface/blob/main/LICENSE)
- [MiniFASNet Apache-2.0 license](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/LICENSE)
